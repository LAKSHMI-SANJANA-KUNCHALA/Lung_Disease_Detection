# # app.py
# """
# Streamlit app: upload -> auto rib-suppression -> choose disease -> classify.

# This version:
#  - tries keras.models.load_model(path) first (full saved model),
#  - if loading fails, it imports model_def.py and uses a constructor
#    (resnet_bs / pneumonia_model / nrds_model / covid19_model / lung_opacity_model)
#    then calls model.load_weights(path) to support weights-only .h5 files.
#  - uses use_container_width=True (replaces deprecated use_column_width).
# """

# import streamlit as st
# import numpy as np
# from pathlib import Path
# from PIL import Image
# import io
# import cv2
# import importlib
# import traceback
# from tensorflow import keras

# st.set_page_config(page_title="Auto rib-suppression → Classify", layout="centered")

# # Default filenames (edit if you want)
# FILES = {
#     "rib": "resnet_bs.h5",
#     "pneumonia": "pneumonia_classification.h5",
#     "nrds": "nrds_classification.h5",
#     "covid_19": "covid19_classification.h5",
#     "lung_opacity": "lung_opacity_classification.h5",
# }

# # Sidebar
# out_dir = st.sidebar.text_input("Save outputs folder", "suppressed_outputs")
# save_outputs = st.sidebar.checkbox("Save outputs (suppressed + prediction)", value=True)

# # Helpers
# def load_and_preprocess_from_pil(pil_img):
#     """Return (orig_uint8_gray, model_input) where model_input is (1,256,256,1) float32 scaled 0..1"""
#     arr = np.array(pil_img.convert("L"))             # H,W uint8
#     orig = arr.copy()
#     resized = cv2.resize(arr, (256,256), interpolation=cv2.INTER_AREA)
#     norm = resized.astype("float32") / 255.0
#     norm = np.expand_dims(norm, axis=-1)             # 256,256,1
#     norm = np.expand_dims(norm, axis=0)              # 1,256,256,1
#     return orig, norm

# def adapt_input_to_model(model, arr_256_gray):
#     """Adapt a 256x256 gray image (H,W) to model.input_shape (batch dim included)."""
#     input_shape = model.input_shape  # typical: (None,H,W,C)
#     if input_shape and len(input_shape) == 4 and input_shape[1] and input_shape[2]:
#         _, H, W, C = input_shape
#         arr = cv2.resize(arr_256_gray, (W, H), interpolation=cv2.INTER_AREA)
#         if C == 1 or C is None:
#             x = np.expand_dims(arr.astype("float32")/255.0, axis=(0,-1))
#         else:
#             stacked = np.stack([arr]*C, axis=-1).astype("float32")/255.0
#             x = np.expand_dims(stacked, axis=0)
#         return x
#     # fallback
#     return np.expand_dims(arr_256_gray.astype("float32")/255.0, axis=(0,-1))

# # Model loader with fallback to model_def constructors
# def try_load_model_full(path: Path):
#     """Try keras.models.load_model (full model). Raise on failure."""
#     return keras.models.load_model(str(path))

# def try_load_weights_with_constructor(path: Path, ctor_name: str):
#     """
#     Import model_def and call constructor ctor_name(), then load_weights(path).
#     """
#     try:
#         model_def = importlib.import_module("model_def")
#         importlib.reload(model_def)
#     except Exception as e:
#         raise RuntimeError("Could not import model_def.py. Create it next to app.py with required constructors.") from e

#     if not hasattr(model_def, ctor_name):
#         raise RuntimeError(f"model_def.py does not define function '{ctor_name}()'.")
#     constructor = getattr(model_def, ctor_name)
#     model = constructor()
#     model.load_weights(str(path))
#     return model

# @st.cache_resource(show_spinner=False)
# def load_model_flex(path_str: str, ctor_name: str = None):
#     p = Path(path_str)
#     if not p.exists():
#         raise FileNotFoundError(f"Model file not found: {p}")
#     # 1) try load full model
#     try:
#         model = try_load_model_full(p)
#         return model
#     except Exception:
#         # 2) fallback to weights-only constructor if provided
#         if not ctor_name:
#             tb = traceback.format_exc()
#             raise RuntimeError("Failed to load full model and no constructor specified for weights-only fallback.\n" + tb)
#         try:
#             model = try_load_weights_with_constructor(p, ctor_name)
#             return model
#         except Exception as e2:
#             tb = traceback.format_exc()
#             raise RuntimeError(f"Failed to load weights into constructor '{ctor_name}': {e2}\n{tb}")

# # Lazy cache for models
# models_cache = {"rib": None, "pneumonia": None, "nrds": None, "covid_19": None, "lung_opacity": None}

# # UI
# st.title("Upload → Auto rib-suppression → Pick disease → Classify")
# st.write("Upload a single chest X-ray (jpg/png). Ribs will be suppressed automatically.")

# uploaded = st.file_uploader("Upload X-ray image", type=["png","jpg","jpeg"])

# if uploaded is not None:
#     try:
#         pil_img = Image.open(io.BytesIO(uploaded.read()))
#     except Exception as e:
#         st.error("Could not read uploaded image.")
#         st.exception(e)
#         st.stop()

#     st.subheader("Original")
#     st.image(pil_img, use_container_width=True)

#     # preprocess for suppression
#     try:
#         orig_uint8, model_input = load_and_preprocess_from_pil(pil_img)
#     except Exception as e:
#         st.error("Preprocessing error.")
#         st.exception(e)
#         st.stop()

#     # load rib model lazily (try full-model first, else weights-only via resnet_bs)
#     if models_cache["rib"] is None:
#         try:
#             with st.spinner("Loading rib-suppression model..."):
#                 models_cache["rib"] = load_model_flex(FILES["rib"], ctor_name="resnet_bs")
#             st.success("Rib-suppression model loaded.")
#         except Exception as e:
#             st.error("Could not load rib-suppression model. If it is weights-only, ensure model_def.resnet_bs() matches the architecture.")
#             st.exception(e)
#             st.stop()

#     # run suppression
#     with st.spinner("Running rib-suppression..."):
#         pred = models_cache["rib"].predict(model_input)
#     pred_img = np.squeeze(pred, axis=0)
#     if pred_img.ndim == 3 and pred_img.shape[2] == 1:
#         pred_img = pred_img[:,:,0]
#     pred_uint8 = (np.clip(pred_img, 0.0, 1.0) * 255.0).astype("uint8")
#     orig_h, orig_w = orig_uint8.shape
#     pred_back = cv2.resize(pred_uint8, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

#     st.subheader("Rib-suppressed")
#     st.image(pred_back, use_container_width=True)

#     # save suppressed image if desired
#     base = Path(uploaded.name).stem
#     if save_outputs:
#         outp = Path(out_dir)
#         outp.mkdir(parents=True, exist_ok=True)
#         supp_path = outp / f"{base}_suppressed.png"
#         Image.fromarray(pred_back).save(str(supp_path))
#         st.info(f"Suppressed image saved → {supp_path}")

#     # disease menu + classify button
#     disease = st.selectbox("Choose disease to classify", ["pneumonia", "nrds", "covid_19", "lung_opacity"])
#     if st.button("Classify"):
#         # load classifier lazily (try full-model first then weights-only)
#         ctor_map = {"pneumonia":"pneumonia_model","nrds":"nrds_model","covid_19":"covid19_model","lung_opacity":"lung_opacity_model"}
#         if models_cache[disease] is None:
#             try:
#                 with st.spinner(f"Loading {disease} classifier..."):
#                     models_cache[disease] = load_model_flex(FILES[disease], ctor_name=ctor_map[disease])
#                 st.success(f"{disease} classifier loaded.")
#             except Exception as e:
#                 st.error(f"Could not load {disease} classifier. If it is weights-only, ensure model_def has a matching constructor.")
#                 st.exception(e)
#                 st.stop()

#         # prepare classifier input from the original (256x256)
#         resized_256 = cv2.resize(orig_uint8, (256,256), interpolation=cv2.INTER_AREA)
#         cls_input = adapt_input_to_model(models_cache[disease], resized_256)

#         with st.spinner("Running classifier..."):
#             pred_cls = models_cache[disease].predict(cls_input)

#         pred_cls = np.array(pred_cls)
#         if pred_cls.ndim == 2 and pred_cls.shape[1] == 1:
#             prob = float(pred_cls[0,0])
#             label = "ABNORMAL" if prob >= 0.5 else "NORMAL"
#             st.markdown(f"**Prediction:** {label}  \n**Confidence:** {prob:.3f}")
#         elif pred_cls.ndim == 2 and pred_cls.shape[1] > 1:
#             probs = pred_cls[0]
#             idx = int(np.argmax(probs))
#             st.markdown(f"**Predicted class index:** {idx}  \n**Confidence:** {probs[idx]:.3f}")
#         else:
#             st.write("Raw output:", pred_cls)

#         if save_outputs:
#             outp = Path(out_dir)
#             npy_path = outp / f"{base}_{disease}_pred.npy"
#             np.save(str(npy_path), pred_cls)
#             st.info(f"Saved classifier output → {npy_path}")
# app.py
"""
Streamlit app: upload -> auto rib-suppression -> choose disease -> classify -> XAI (Grad-CAM)

Assumptions:
 - Classifier .h5 files are full saved Keras models (model.save(...)) or weights+constructor via model_def.py (optional).
 - Rib-suppression model may be weights-only; loader has fallback if you provide constructors (resnet_bs etc).
 - This file contains a robust Grad-CAM implementation adapted for variable model shapes.

Run:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image
import io
import cv2
import importlib
import traceback
from tensorflow import keras
import tensorflow as tf

st.set_page_config(page_title="Lung Disease Detection", layout="centered")

# ------- filenames (edit if needed) -------
FILES = {
    "rib": "resnet_bs.h5",
    "pneumonia": "pneumonia_classification.h5",
    "nrds": "nrds_classification.h5",
    "covid_19": "covid19_classification.h5",
    "lung_opacity": "lung_opacity_classification.h5",
}

# sidebar options
out_dir = st.sidebar.text_input("Save outputs folder", "suppressed_outputs")
save_outputs = st.sidebar.checkbox("Save outputs (suppressed + prediction + xai)", value=True)
show_xai = st.sidebar.checkbox("Enable XAI (Grad-CAM) after classification", value=True)

# ---------------- helpers ----------------
def load_and_preprocess_from_pil(pil_img):
    """Return (orig_uint8_gray, model_input) where model_input is (1,256,256,1) float32 scaled 0..1"""
    arr = np.array(pil_img.convert("L"))             # H,W uint8
    orig = arr.copy()
    resized = cv2.resize(arr, (256,256), interpolation=cv2.INTER_AREA)
    norm = resized.astype("float32") / 255.0
    norm = np.expand_dims(norm, axis=-1)             # 256,256,1
    norm = np.expand_dims(norm, axis=0)              # 1,256,256,1
    return orig, norm

def adapt_input_to_model(model, arr_256_gray):
    """Adapt a 256x256 gray image (H,W) to model.input_shape (batch dim included)."""
    input_shape = model.input_shape  # typical: (None,H,W,C)
    if input_shape and len(input_shape) == 4 and input_shape[1] and input_shape[2]:
        _, H, W, C = input_shape
        arr = cv2.resize(arr_256_gray, (W, H), interpolation=cv2.INTER_AREA)
        if C == 1 or C is None:
            x = np.expand_dims(arr.astype("float32")/255.0, axis=(0,-1))
        else:
            stacked = np.stack([arr]*C, axis=-1).astype("float32")/255.0
            x = np.expand_dims(stacked, axis=0)
        return x
    # fallback: return (1,256,256,1)
    return np.expand_dims(arr_256_gray.astype("float32")/255.0, axis=(0,-1))

# ------------- model loading -------------
def try_load_model_full(path: Path):
    return keras.models.load_model(str(path))

def try_load_weights_with_constructor(path: Path, ctor_name: str):
    try:
        model_def = importlib.import_module("model_def")
        importlib.reload(model_def)
    except Exception as e:
        raise RuntimeError("Could not import model_def.py. Create it next to app.py with required constructors.") from e
    if not hasattr(model_def, ctor_name):
        raise RuntimeError(f"model_def.py does not define function '{ctor_name}()'.")
    constructor = getattr(model_def, ctor_name)
    model = constructor()
    model.load_weights(str(path))
    return model

@st.cache_resource(show_spinner=False)
def load_model_flex(path_str: str, ctor_name: str = None):
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    # try full model
    try:
        model = try_load_model_full(p)
        return model
    except Exception:
        if not ctor_name:
            tb = traceback.format_exc()
            raise RuntimeError("Failed to load full model and no constructor specified for weights-only fallback.\n" + tb)
        try:
            model = try_load_weights_with_constructor(p, ctor_name)
            return model
        except Exception as e2:
            tb = traceback.format_exc()
            raise RuntimeError(f"Failed to load weights into constructor '{ctor_name}': {e2}\n{tb}")

# lazy cache
models_cache = {"rib": None, "pneumonia": None, "nrds": None, "covid_19": None, "lung_opacity": None}

# ------------- Grad-CAM utils -------------
def find_last_conv_layer(model):
    """
    Attempt to find the last convolutional (4D output) layer in model.
    Prefer layers with 'conv' in name, otherwise pick last layer with 4D output.
    """
    for layer in reversed(model.layers):
        # some layers may not have output_shape before build; use try/except
        try:
            out_shape = layer.output_shape
        except Exception:
            out_shape = None
        name = layer.name.lower()
        if out_shape and len(out_shape) == 4:
            # prefer conv-like names
            if "conv" in name or "block" in name or "features" in name:
                return layer.name
    # fallback: pick the last with 4D output
    for layer in reversed(model.layers):
        try:
            out_shape = layer.output_shape
        except Exception:
            out_shape = None
        if out_shape and len(out_shape) == 4:
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    img_array: preprocessed input matching model input shape (batch dim included)
    model: keras model
    last_conv_layer_name: string
    pred_index: class index or None (use top predicted)
    Returns: heatmap (H,W) normalized 0..1
    """
    grad_model = None
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except Exception as e:
        raise RuntimeError(f"Could not fetch layer '{last_conv_layer_name}' from model: {e}")

    # Build a model that maps the input to the activations of the last conv layer
    # and the model's predictions
    grad_model = keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        tape.watch(inputs)
        conv_outputs, predictions = grad_model(inputs)
        if pred_index is None:
            # prediction may be shape (1,1) or (1,N)
            if predictions.shape[-1] == 1:
                pred_index = 0
            else:
                pred_index = tf.argmax(predictions[0])
        # pick the score for pred_index (works for both binary sigmoid and multi-class softmax)
        if predictions.shape[-1] == 1:
            # sigmoid binary: treat scalar as score
            score = predictions[:, 0]
        else:
            # multi-class: take the logit/prob at pred_index
            score = predictions[:, pred_index]

    # compute gradients of the score w.r.t. conv outputs
    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients are None (check model and input).")

    # pooled grads across the spatial dimensions: shape (channels,)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))  # (channels,)

    # conv_outputs has shape (batch,H,W,channels); take the first batch
    conv_outputs = conv_outputs[0]  # (H,W,channels)
    # ensure same dtype
    conv_outputs = tf.cast(conv_outputs, tf.float32)
    pooled_grads = tf.cast(pooled_grads, tf.float32)

    # Multiply each channel in feature map array by "how important this channel is"
    # Use broadcasting: pooled_grads shape => (channels,) -> [None,None,channels] -> multiplies with (H,W,channels)
    conv_outputs *= pooled_grads[tf.newaxis, tf.newaxis, :]

    # The channel-wise mean of the resulting feature map is the heatmap
    heatmap = tf.reduce_sum(conv_outputs, axis=-1)

    # Relu and normalize
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap.numpy())
    heatmap /= max_val
    return heatmap.numpy()

def overlay_heatmap_on_image(heatmap, original_image_uint8, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    heatmap: HxW float 0..1
    original_image_uint8: HxW (grayscale) or HxWx3 (rgb)
    returns overlay uint8 RGB
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)  # BGR
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # if original is grayscale expand to 3 channels
    if original_image_uint8.ndim == 2:
        orig_rgb = np.stack([original_image_uint8]*3, axis=-1)
    else:
        orig_rgb = original_image_uint8

    heatmap_resized = cv2.resize(heatmap_color, (orig_rgb.shape[1], orig_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(orig_rgb.astype("uint8"), 1.0 - alpha, heatmap_resized.astype("uint8"), alpha, 0)
    return overlay

# ------------- UI -------------
st.title("Upload → Auto rib-suppression → Choose disease → Classify → XAI")
st.write("Upload a chest X-ray. Ribs will be suppressed automatically, then choose disease and optionally run Grad-CAM XAI.")

uploaded = st.file_uploader("Upload X-ray image", type=["png","jpg","jpeg"])

if uploaded is not None:
    try:
        pil_img = Image.open(io.BytesIO(uploaded.read()))
    except Exception as e:
        st.error("Could not read uploaded image.")
        st.exception(e)
        st.stop()

    st.subheader("Original")
    st.image(pil_img, use_container_width=True)

    # preprocess for suppression
    try:
        orig_uint8, model_input = load_and_preprocess_from_pil(pil_img)
    except Exception as e:
        st.error("Preprocessing error.")
        st.exception(e)
        st.stop()

    # load rib model lazily
    if models_cache["rib"] is None:
        try:
            with st.spinner("Loading rib-suppression model..."):
                models_cache["rib"] = load_model_flex(FILES["rib"], ctor_name="resnet_bs")
            st.success("Rib-suppression model loaded.")
        except Exception as e:
            st.error("Could not load rib-suppression model. If it is weights-only, ensure model_def.resnet_bs() matches the architecture.")
            st.exception(e)
            st.stop()

    # run suppression
    with st.spinner("Running rib-suppression..."):
        pred = models_cache["rib"].predict(model_input)
    pred_img = np.squeeze(pred, axis=0)
    if pred_img.ndim == 3 and pred_img.shape[2] == 1:
        pred_img = pred_img[:,:,0]
    pred_uint8 = (np.clip(pred_img, 0.0, 1.0) * 255.0).astype("uint8")
    orig_h, orig_w = orig_uint8.shape
    pred_back = cv2.resize(pred_uint8, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

    st.subheader("Rib-suppressed")
    st.image(pred_back, use_container_width=True)

    # save suppressed image if desired
    base = Path(uploaded.name).stem
    if save_outputs:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        supp_path = outp / f"{base}_suppressed.png"
        Image.fromarray(pred_back).save(str(supp_path))
        st.info(f"Suppressed image saved → {supp_path}")

    # classification UI
    disease = st.selectbox("Choose disease to classify", ["pneumonia", "nrds", "covid_19", "lung_opacity"])
    if st.button("Classify"):
        ctor_map = {"pneumonia":"pneumonia_model","nrds":"nrds_model","covid_19":"covid19_model","lung_opacity":"lung_opacity_model"}
        if models_cache[disease] is None:
            try:
                with st.spinner(f"Loading {disease} classifier..."):
                    models_cache[disease] = load_model_flex(FILES[disease], ctor_name=ctor_map[disease])
                st.success(f"{disease} classifier loaded.")
            except Exception as e:
                st.error(f"Could not load {disease} classifier. If it is weights-only, ensure model_def has a matching constructor.")
                st.exception(e)
                st.stop()

        # prepare classifier input from the original (256x256)
        resized_256 = cv2.resize(orig_uint8, (256,256), interpolation=cv2.INTER_AREA)
        cls_input = adapt_input_to_model(models_cache[disease], resized_256)

        with st.spinner("Running classifier..."):
            pred_cls = models_cache[disease].predict(cls_input)

        pred_cls = np.array(pred_cls)
        if pred_cls.ndim == 2 and pred_cls.shape[1] == 1:
            prob = float(pred_cls[0,0])
            label = "ABNORMAL" if prob >= 0.5 else "NORMAL"
            st.markdown(f"**Prediction:** {label}  \n**Confidence:** {prob:.3f}")
        elif pred_cls.ndim == 2 and pred_cls.shape[1] > 1:
            probs = pred_cls[0]
            idx = int(np.argmax(probs))
            st.markdown(f"**Predicted class index:** {idx}  \n**Confidence:** {probs[idx]:.3f}")
        else:
            st.write("Raw output:", pred_cls)

        # Save classifier output
        if save_outputs:
            outp = Path(out_dir)
            npy_path = outp / f"{base}_{disease}_pred.npy"
            np.save(str(npy_path), pred_cls)
            st.info(f"Saved classifier output → {npy_path}")

        # XAI: Grad-CAM (optional)
        if show_xai:
            try:
                with st.spinner("Preparing Grad-CAM..."):
                    # model may expect 3-channel RGB at some input size (e.g. VGG16). Build a gradcam input that matches model.input_shape.
                    model_for_gradcam = models_cache[disease]
                    # prepare a single input matching model input shape
                    input_shape = model_for_gradcam.input_shape
                    # create gradcam_input from the resized_256 gray image but adapt
                    gradcam_input = adapt_input_to_model(model_for_gradcam, resized_256)  # batch included
                    # find last conv layer name
                    last_conv = find_last_conv_layer(model_for_gradcam)
                    if last_conv is None:
                        st.warning("No conv layer found for Grad-CAM in this model; cannot compute heatmap.")
                    else:
                        # If model expects different input size (e.g. 224x224,3), adapt_input_to_model already took care of it.
                        # Choose target index: for binary sigmoid, use index 0; for multi-class, use argmax
                        if pred_cls.ndim == 2 and pred_cls.shape[1] == 1:
                            target_index_for_xai = None  # will use the scalar prediction
                        else:
                            target_index_for_xai = int(np.argmax(pred_cls[0]))

                        heatmap = make_gradcam_heatmap(gradcam_input, model_for_gradcam, last_conv, pred_index=target_index_for_xai)
                        overlay = overlay_heatmap_on_image(heatmap, orig_uint8, alpha=0.4)
                        st.subheader("Grad-CAM (overlay)")
                        st.image(overlay, use_container_width=True)

                        if save_outputs:
                            heatmap_path = outp / f"{base}_{disease}_gradcam.png"
                            Image.fromarray(overlay).save(str(heatmap_path))
                            st.info(f"Saved Grad-CAM overlay → {heatmap_path}")

            except Exception as e:
                st.error("Error computing Grad-CAM.")
                st.exception(e)

st.caption("Notes: Grad-CAM expects the classifier to be a Keras model. Large models may be slow; GPU recommended.")
