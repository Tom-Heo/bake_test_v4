import os
import io
import torch
import torch.nn.functional as F
import numpy as np
import imageio.v3 as iio
import cv2  # [필수] 16-bit PNG 저장을 위해 OpenCV 추가
import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

# Local Modules
from config import Config
from core.net import BakeNet
from core.palette import Palette

# -----------------------------------------------------------------------------
# [1] Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bake - Accurate Color Restoration",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Dark Minimalist Theme
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Brand Color: #D41201 */
    .highlight {
        color: #D41201;
        font-weight: 600;
    }
    
    /* Button Style */
    div.stButton > button {
        background-color: #D41201;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: background-color 0.2s;
    }
    div.stButton > button:hover {
        background-color: #FF1F0C;
        border-color: #FF1F0C;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# [2] Model Loading (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load BakeNet and Converter Modules"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Model
    model = BakeNet(dim=Config.INTERNAL_DIM).to(device)

    # Load Checkpoint
    ckpt_path = Config.LAST_CKPT_PATH

    if not os.path.exists(ckpt_path):
        return None, None, None, device

    try:
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Load EMA if available (Preferred for inference)
        if "ema_shadow" in checkpoint:
            model.load_state_dict(checkpoint["ema_shadow"], strict=False)
        else:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        model.eval()

        # Converters
        to_oklabp = Palette.sRGBtoOklabP().to(device)
        to_rgb = Palette.OklabPtosRGB().to(device)

        return model, to_oklabp, to_rgb, device

    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        return None, None, None, device


# 전역 로드 (모델이 없으면 UI에서 처리)
model, to_oklabp, to_rgb, device = load_model()


# -----------------------------------------------------------------------------
# [3] Processing Logic
# -----------------------------------------------------------------------------
def process_image(uploaded_file, bit_depth_option, model, to_oklabp, to_rgb, device):
    """
    [Inference Pipeline]
    Load -> Normalize -> Upscale(x2 Nearest) -> Pad -> BakeNet -> Unpad -> Output
    """
    bytes_data = uploaded_file.getvalue()
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1]

    # 1. Read Image
    try:
        # imageio로 포맷 자동 감지하여 읽기
        img_np = iio.imread(bytes_data, extension=ext)
    except:
        try:
            # 실패 시 확장자 없이 바이트 스트림으로 시도
            img_np = iio.imread(bytes_data)
        except Exception as e:
            return None, None, f"Error: {e}"

    # Handle Alpha Channel (Drop it)
    if img_np.ndim == 3 and img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]

    img_float = img_np.astype(np.float32)
    detected_msg = ""

    # 2. Normalize based on Bit Depth
    if img_np.dtype == np.uint16:
        # 16-bit Container Logic
        if bit_depth_option == "Auto":
            max_val = img_np.max()
            if max_val <= 1023:
                depth = 10
            elif max_val <= 4095:
                depth = 12
            elif max_val <= 16383:
                depth = 14
            else:
                depth = 16
            detected_msg = f"Auto-detected: {depth}-bit"
        else:
            depth = int(bit_depth_option)
            detected_msg = f"Manual Override: {depth}-bit"

        img_float = img_float / ((2**depth) - 1)

    elif img_np.dtype == np.uint8:
        # 8-bit
        img_float = img_float / 255.0
        detected_msg = "8-bit Source"

    # To Tensor (HWC -> CHW)
    input_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)
    input_tensor = input_tensor.clamp(0.0, 1.0)

    # -------------------------------------------------------------------------
    # [Upscaling] x2 Nearest Neighbor
    # -------------------------------------------------------------------------
    # BakeNet 입력 전에 2배 확대를 수행하여, 결과물 해상도도 2배가 되도록 함.
    input_tensor = F.interpolate(input_tensor, scale_factor=2, mode="bilinear")
    detected_msg += " | x2 Upscaled"
    # -------------------------------------------------------------------------

    # 3. Padding (Reflect for even dims)
    # Upscaling 이후의 해상도를 기준으로 패딩 계산
    _, _, h, w = input_tensor.shape
    pad_h = 1 if (h % 2 != 0) else 0
    pad_w = 1 if (w % 2 != 0) else 0

    if pad_h + pad_w > 0:
        input_padded = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode="reflect")
    else:
        input_padded = input_tensor

    # 4. Inference
    with torch.no_grad():
        input_oklabp = to_oklabp(input_padded)
        output_oklabp = model(input_oklabp)
        output_rgb = to_rgb(output_oklabp)

    # 5. Unpad & Clamp
    # 패딩된 부분 잘라내기
    output_rgb = output_rgb[:, :, :h, :w].clamp(0.0, 1.0)

    # input_tensor 역시 2배 확대된 상태로 반환하여 비교 슬라이더의 1:1 매칭 보장
    return input_tensor.cpu(), output_rgb.cpu(), detected_msg


def to_display_image(tensor):
    """
    [For Web Display]
    Tensor(0~1) -> Numpy(0~255 uint8)
    웹 브라우저는 16비트를 표시 못 하므로 8비트로 변환
    """
    img = tensor.squeeze(0).permute(1, 2, 0).numpy()
    return (img * 255.0).astype(np.uint8)


def to_download_bytes(tensor):
    """
    [For Download]
    Tensor(0~1) -> 16-bit PNG Bytes
    사용자에게는 무손실 16비트 결과물을 제공 (OpenCV 사용)
    """
    # 1. Tensor (1, 3, H, W) -> Numpy (H, W, 3)
    img_np = tensor.squeeze(0).permute(1, 2, 0).numpy()

    # 2. RGB -> BGR 변환 (OpenCV는 BGR 순서를 사용)
    img_bgr = img_np[..., ::-1]

    # 3. Scale to 16-bit Integer
    img_uint16 = (img_bgr * 65535.0).astype(np.uint16)

    # 4. Encode to PNG using OpenCV
    is_success, buffer = cv2.imencode(".png", img_uint16)

    if not is_success:
        return None

    # 5. Return as BytesIO
    # buffer는 numpy array 형태이므로 tobytes()로 바이트 변환
    return io.BytesIO(buffer.tobytes())


# -----------------------------------------------------------------------------
# [4] UI Layout
# -----------------------------------------------------------------------------

# Sidebar Controls
with st.sidebar:
    st.header("Bake Settings")

    st.markdown("---")
    st.subheader("Input Bit Depth")
    bit_option = st.selectbox(
        "Source Processing Mode",
        ["Auto", "10", "12", "14", "16", "8"],
        index=0,
        help="Use 'Auto' for most cases. Manually select if your dark 10/12-bit footage is misdetected.",
    )

    if bit_option == "Auto":
        st.info("💡 Auto-detects bit depth based on pixel intensity.")
    else:
        st.warning(f"⚠️ Forcing {bit_option}-bit interpretation.")

    st.markdown("---")
    st.markdown("### System Status")
    if model is not None:
        st.success("🟢 BakeNet Loaded")
        st.caption(f"Running on: {device}")
    else:
        st.error("🔴 Model Not Found")
        st.caption("Please place 'last.pth' in checkpoints/")


# Main Area
st.title("Bake")
st.markdown("#### Accurate, therefore beautiful.")
st.markdown("AI-Powered 10-bit/12-bit Color Restoration (with x2 Upscaling)")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload Frame (PNG, TIFF, JPG, DPX)",
    type=["png", "jpg", "jpeg", "tif", "tiff", "dpx"],
)

if uploaded_file is not None:
    if model is None:
        st.error("BakeNet model is not loaded. Cannot process image.")
    else:
        # Processing Indicator
        with st.spinner("Baking... Recovering lost gradients and details."):

            # Run Pipeline
            input_t, output_t, msg = process_image(
                uploaded_file, bit_option, model, to_oklabp, to_rgb, device
            )

            if input_t is not None:
                st.success(f"Processing Complete! {msg}")
                st.markdown("---")

                # A. Comparison Slider
                st.subheader("Before / After Comparison")
                st.caption("Slide to see the restored gradients (x2 Upscaled View)")

                img_before = to_display_image(input_t)
                img_after = to_display_image(output_t)

                # Image Comparison Component
                image_comparison(
                    img1=img_before,
                    img2=img_after,
                    label1="Original (x2 Bilinear)",
                    label2="Baked (Restored)",
                    width=700,
                    starting_position=50,
                    show_labels=True,
                    make_responsive=True,
                    in_memory=True,
                )

                st.markdown("---")

                # B. Download Section
                st.subheader("Export Result")

                # 컬럼을 나누어 원본(Upscaled)과 결과물 모두 다운로드 가능하게 배치
                col1, col2, col3 = st.columns([1, 1, 2])

                with col1:
                    st.download_button(
                        label="⬇️ Download Output",
                        data=to_download_bytes(output_t),
                        file_name="bake_result_16bit.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                with col2:
                    st.download_button(
                        label="⬇️ Download Input",
                        data=to_download_bytes(input_t),
                        file_name="bake_input_nearest_16bit.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                with col3:
                    st.info(
                        "**Professional Export:** Both files are **16-bit PNGs** (x2 Upscaled). "
                        "'Input' is Nearest-Neighbor scaled, 'Output' is BakeNet restored."
                    )

            else:
                st.error(f"Processing Failed: {msg}")

else:
    # Landing Page Description
    st.info("👋 Welcome! Upload a frame to start restoring colors.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🎨 Oklab Precision**")
        st.caption(
            "Perceptually accurate color handling exactly as the human eye sees."
        )

    with col2:
        st.markdown("**🔧 10/12-bit Support**")
        st.caption("Designed for professional Log footage and high-bit workflows.")

    with col3:
        st.markdown("**✨ Structure Recovery**")
        st.caption("Removes banding artifacts and restores smooth gradients.")

# Footer
st.markdown("---")
st.caption("Bake v4 | Research & Production Code | Developed for Creators")
