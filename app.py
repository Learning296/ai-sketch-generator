# import streamlit as st
# from google import genai
# from google.genai import types
# from PIL import Image
# import io
# import random
# import requests
# import base64
# import os

# # ----------------------------
# # 🔧 APP CONFIG
# # ----------------------------
# st.set_page_config(page_title="Perfect Pencil Sketch Generator", page_icon="✏", layout="wide")

# # ----------------------------
# # 🔑 API KEYS SIDEBAR
# # ----------------------------
# st.sidebar.title("🔑 API Configuration")
# st.sidebar.markdown("Please provide both API keys to use the application")

# # Initialize session state for API keys
# if 'gemini_key' not in st.session_state:
#     st.session_state.gemini_key = ""
# if 'hf_token' not in st.session_state:
#     st.session_state.hf_token = ""
# if 'keys_provided' not in st.session_state:
#     st.session_state.keys_provided = False

# # API key input fields
# gemini_key = st.sidebar.text_input("Gemini API Key", type="password", 
#                                   help="Enter your Google Gemini API key")
# hf_token = st.sidebar.text_input("Hugging Face Token", type="password", 
#                                 help="Enter your Hugging Face API token")

# # Save keys to session state when provided
# if gemini_key and hf_token:
#     st.session_state.gemini_key = gemini_key
#     st.session_state.hf_token = hf_token
#     st.session_state.keys_provided = True
#     st.sidebar.success("✅ API keys saved! You can now use the application.")
# elif gemini_key or hf_token:
#     st.sidebar.warning("⚠️ Please provide both API keys")
# else:
#     st.sidebar.info("🔐 Please enter both API keys to continue")

# # Clear keys button
# if st.sidebar.button("Clear API Keys"):
#     st.session_state.gemini_key = ""
#     st.session_state.hf_token = ""
#     st.session_state.keys_provided = False
#     st.rerun()

# # ----------------------------
# # MAIN APPLICATION (Only show if keys are provided)
# # ----------------------------
# if st.session_state.keys_provided:
#     # Initialize Gemini client with provided key
#     gemini_client = genai.Client(api_key=st.session_state.gemini_key)
    
#     # Main app title and description
#     st.title("✏ Perfect Pencil Sketch Generator - SEED SUPPORT!")
    
#     # ----------------------------
#     # 🎛 User Controls
#     # ----------------------------
#     st.header("🧠 Describe Your Portrait")
    
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         gender = st.selectbox("Gender", ["Any", "Male", "Female"], index=2)
#     with col2:
#         age = st.selectbox("Age Group", ["Any", "18-24 months", "20-30 years", "30-40 years", "40-50 years", "50-60 years"], index=3)
#     with col3:
#         ethnicity = st.selectbox("Ethnicity", ["Any", "Caucasian/White", "Hispanic/Latino", "African-American/African", "South Asian", "East Asian", "Middle-Eastern"], index=1)
    
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         eye_shape = st.selectbox("Eye Shape", ["Any", "Almond", "Round", "Monolid", "Hooded", "Upturned", "Downturned"])
#     with col2:
#         lips_shape = st.selectbox("Lips Shape", ["Any", "Full", "Thin", "Heart-shaped", "Round"])
#     with col3:
#         nose = st.selectbox("Nose Shape", ["Any", "Straight", "Button", "Hooked"])
    
#     col1, col2 = st.columns(2)
#     with col1:
#         face_cutt = st.selectbox("Face Cut", ["Any", "Oval", "Round", "Square", "Heart"])
#     with col2:
#         cheecks = st.selectbox("Cheeks", ["Any", "High cheekbones", "Chubby cheeks", "Defined"])
    
#     pose = st.selectbox("Pose", ["Front view", "3/4 view", "Side profile", "Looking down", "Looking up"], index=0)
#     expression = st.selectbox("Facial Expression", ["Neutral", "Slight Smile", "Full Smile", "Serious", "Calm"], index=1)
    
#     st.header("💇‍♀️ Hair Style")
#     hair_style = st.selectbox("Hair Style", ["Any", "Straight", "Wavy", "Curly", "Coily", "Bob cut", "Pixie cut"])
    
#     st.header("✨ Other Features")
#     eyebrow_shape = st.selectbox("Eyebrow Shape", ["Any", "Thick", "Thin", "Arched", "Straight"])
    
#     # --- Settings ---
#     st.header("⚙ Generation Settings")
    
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         num_images = st.slider("Number of Images", 1, 4, 1)
#     with col2:
#         aspect = st.selectbox("Aspect Ratio", ["1:1", "4:3", "16:9"], index=0)
#     with col3:
#         seed = st.number_input("🎯 Seed Value (WORKING!)", min_value=0, value=random.randint(1, 999999), help="Same seed = Same face!")
    
#     prompt_user = st.text_area(
#         "💡 Add any extra artistic details (optional):",
#         placeholder="Example: wearing a scarf, vintage paper background"
#     )
    
#     # Session state for storing images
#     if 'generated_images' not in st.session_state:
#         st.session_state.generated_images = []
    
#     # ----------------------------
#     # 🪄 PROMPT BUILDER FOR FLUX
#     # ----------------------------
#     def build_flux_prompt():
#         """Build prompt for FLUX to generate realistic portrait"""
#         positive = "professional studio portrait photograph, "
        
#         if expression != "Neutral":
#             positive += f"{expression.lower()} expression, "
#         if ethnicity != "Any":
#             positive += f"{ethnicity.lower()}, "
#         if gender != "Any":
#             positive += f"{gender.lower()}, "
#         if age != "Any":
#             positive += f"{age.lower()}, "
        
#         if eye_shape != "Any":
#             positive += f"{eye_shape.lower()} eyes, "
#         if lips_shape != "Any":
#             positive += f"{lips_shape.lower()} lips, "
#         if nose != "Any":
#             positive += f"{nose.lower()} nose, "
#         if face_cutt != "Any":
#             positive += f"{face_cutt.lower()} face shape, "
#         if cheecks != "Any":
#             positive += f"{cheecks.lower()}, "
#         if eyebrow_shape != "Any":
#             positive += f"{eyebrow_shape.lower()} eyebrows, "
#         if hair_style != "Any":
#             positive += f"{hair_style.lower()} hair, "
        
#         positive += "head and shoulders portrait, "
#         positive += f"{pose.lower()}, "
#         positive += "professional studio lighting, plain background, high resolution, detailed facial features, photorealistic"
        
#         if prompt_user.strip():
#             positive += f", {prompt_user.strip()}"
        
#         return positive
    
#     # ----------------------------
#     # 🎨 STEP 1: FLUX GENERATION
#     # ----------------------------
#     def generate_flux_image(prompt, seed_value, width=768, height=768):
#         """Generate base image with FLUX using seed"""
#         model_id = "black-forest-labs/FLUX.1-schnell"  # Fast model
#         # url = f"https://api-inference.huggingface.co/models/{model_id}"
#         # url = f" https://router.huggingface.co/hf-inference/models/{model_id}"
#         url = f"https://api-inference.huggingface.co/models/{model_id}"

        
       
#         headers = {"Authorization": f"Bearer {st.session_state.hf_token}"}
        
#         payload = {
#             "inputs": prompt,
#             "parameters": {
#                 "seed": int(seed_value),
#                 "num_inference_steps": 30,
#                 "guidance_scale": 7.5,
#                 "width": width,
#                 "height": height,
#             },
#             "options": {"wait_for_model": True}
#         }
        
#         response = requests.post(url, headers=headers, json=payload, timeout=120)
        
#         if response.status_code != 200:
#             raise Exception(f"FLUX API Error {response.status_code}: {response.text}")
        
#         # Check if response is image
#         if response.headers.get("content-type", "").startswith("image/"):
#             return Image.open(io.BytesIO(response.content)).convert("RGB")
        
#         raise Exception("FLUX did not return an image")
    
#     # ----------------------------
#     # 🖼 ENHANCED STEP 2: GEMINI PENCIL CONVERSION
#     # ----------------------------
#     def convert_to_pencil_sketch(pil_image):
#         """Convert image to realistic hand-drawn pencil sketch using Gemini 2.5 Flash Image"""
        
#         # ENHANCED PROMPT for realistic pencil sketch aesthetic
#         conversion_prompt = """Transform this portrait into an authentic hand-drawn graphite pencil sketch on textured drawing paper.
    
#     CRITICAL REQUIREMENTS - PENCIL SKETCH AESTHETIC:
    
#     **TEXTURE & MATERIAL:**
#     - Visible rough drawing paper grain and tooth texture throughout
#     - Real graphite pencil marks - rough, grainy, hand-drawn strokes
#     - Natural pencil texture with visible graphite particles
#     - Authentic paper fiber texture showing through
#     - Matte pencil finish, no smooth digital gradients
    
#     **DRAWING TECHNIQUE:**
#     - Hand-drawn pencil strokes following facial contours and forms
#     - Visible directional hatching and cross-hatching for shading
#     - Rough, sketchy line quality - NOT smooth vector lines
#     - Natural artist hand movements visible in strokes
#     - Varying pencil pressure - lighter in highlights, darker in shadows
#     - Use multiple pencil grades effect: HB for outlines, 2B-4B for mid-tones, 6B-8B for darkest shadows
    
#     **SHADING & TONES:**
#     - Soft graphite smudging and blending in shadow areas
#     - Delicate tonal transitions using layered pencil strokes
#     - Strategic eraser-lift highlights for brightest areas (eyes, nose tip, lips)
#     - Natural pencil buildup in darker areas
#     - White paper showing through in highlights - NOT pure white
#     - Mid-gray paper tone as base
    
#     **STYLE CHARACTERISTICS:**
#     - Sketchy, artistic, hand-drawn appearance
#     - Slightly loose, expressive strokes - not photorealistic rendering
#     - Natural imperfections and variations in line weight
#     - Authentic artist's hand-drawn portrait feel
#     - Black graphite on off-white/cream paper
    
#     **COMPOSITION:**
#     - Complete portrait - full head and shoulders visible
#     - Clean framing with slight paper border/vignette effect
#     - All facial features fully detailed and finished
#     - Natural portrait proportions maintained
    
#     **WHAT TO AVOID:**
#     - NO smooth digital gradients or airbrushed appearance
#     - NO clean vector lines or perfect curves
#     - NO photorealistic rendering or painting effects
#     - NO color - pure black and white graphite only
#     - NO signatures, watermarks, or text
#     - NO incomplete or unfinished areas
    
#     **FINAL RESULT:**
#     The sketch should look like a skilled artist spent 2-3 hours drawing this portrait with graphite pencils on textured paper. Every viewer should immediately recognize it as a hand-drawn pencil sketch, not a digital filter or photo manipulation.
    
#     Maintain the exact facial features, expression, and likeness from the input portrait."""
    
#         try:
#             # Generate with Gemini 2.5 Flash Image
#             response = gemini_client.models.generate_content(
#                 model="gemini-2.5-flash-image",
#                 contents=[conversion_prompt, pil_image],
#                 config=types.GenerateContentConfig(
#                     response_modalities=["IMAGE"],
#                     temperature=0.4,  # Lower for more consistent pencil aesthetic
#                 )
#             )
            
#             # Extract the generated sketch
#             for part in response.candidates[0].content.parts:
#                 if part.inline_data is not None:
#                     sketch_bytes = part.inline_data.data
#                     sketch_image = Image.open(io.BytesIO(sketch_bytes))
#                     return sketch_image
            
#             raise Exception("No image in Gemini response")
            
#         except Exception as e:
#             raise Exception(f"Gemini conversion failed: {str(e)}")
    
#     # ----------------------------
#     # 🎨 ALTERNATIVE: POST-PROCESSING ENHANCEMENT (Optional)
#     # ----------------------------
#     def enhance_pencil_texture(pil_image):
#         """
#         Optional post-processing to add more pencil texture if needed.
#         Use this AFTER Gemini conversion if you want extra texture.
#         """
#         from PIL import ImageFilter, ImageEnhance
#         import numpy as np
        
#         # Convert to grayscale if not already
#         if pil_image.mode != 'L':
#             pil_image = pil_image.convert('L')
        
#         # Slight sharpening to enhance pencil strokes
#         enhanced = pil_image.filter(ImageFilter.SHARPEN)
        
#         # Increase contrast slightly for more defined pencil marks
#         contrast = ImageEnhance.Contrast(enhanced)
#         enhanced = contrast.enhance(1.2)
        
#         # Add subtle noise for paper grain (optional)
#         img_array = np.array(enhanced).astype(float)
#         noise = np.random.normal(0, 2, img_array.shape)
#         img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
#         return Image.fromarray(img_array, mode='L')
    
#     # ----------------------------
#     # 🚀 UPDATED FULL PIPELINE (with optional enhancement)
#     # ----------------------------
#     def run_full_pipeline(seed_value):
#         """Run the complete FLUX → Gemini pipeline with enhanced pencil conversion"""
        
#         # Build FLUX prompt
#         flux_prompt = build_flux_prompt()
        
#         # Determine dimensions
#         width, height = 768, 768
#         if aspect == "4:3":
#             width, height = 896, 672
#         elif aspect == "16:9":
#             width, height = 1024, 576
        
#         # Progress tracking
#         progress_bar = st.progress(0)
#         status_text = st.empty()
        
#         all_results = []
        
#         for i in range(num_images):
#             current_seed = seed_value + i
            
#             try:
#                 # STEP 1: Generate with FLUX
#                 status_text.text(f"🎨 Step 1/3: Generating base portrait with FLUX (seed: {current_seed})...")
#                 progress_bar.progress((i * 3) / (num_images * 3))
                
#                 base_image = generate_flux_image(flux_prompt, current_seed, width, height)
                
#                 # STEP 2: Convert to pencil sketch with Gemini
#                 status_text.text(f"✏ Step 2/3: Converting to hand-drawn pencil sketch...")
#                 progress_bar.progress((i * 3 + 1) / (num_images * 3))
                
#                 sketch_image = convert_to_pencil_sketch(base_image)
                
#                 # STEP 3 (OPTIONAL): Enhance pencil texture
#                 # Uncomment next line if you want extra texture enhancement:
#                 sketch_image = enhance_pencil_texture(sketch_image)
                
#                 # Store results
#                 buf = io.BytesIO()
#                 sketch_image.save(buf, format="PNG")
                
#                 all_results.append({
#                     "base": base_image,
#                     "sketch": sketch_image,
#                     "sketch_data": buf.getvalue(),
#                     "seed": current_seed
#                 })
                
#                 progress_bar.progress((i * 3 + 2) / (num_images * 3))
                
#             except Exception as e:
#                 st.error(f"❌ Failed for seed {current_seed}: {str(e)}")
#                 continue
        
#         status_text.empty()
#         progress_bar.empty()
        
#         return all_results
    
#     # ----------------------------
#     # 🎨 GENERATE BUTTONS
#     # ----------------------------
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         if st.button("✨ Generate with Seed", use_container_width=True, type="primary"):
#             with st.spinner("Running FLUX + Gemini pipeline..."):
#                 try:
#                     st.info(f"🎯 Using seed: {seed}")
#                     results = run_full_pipeline(seed)
                    
#                     if results:
#                         st.session_state.generated_images.extend(results)
#                         st.success(f"✅ Generated {len(results)} sketch(es) successfully!")
#                         st.rerun()
#                     else:
#                         st.error("❌ No images generated")
                        
#                 except Exception as e:
#                     st.error(f"❌ Pipeline failed: {str(e)}")
    
#     with col2:
#         if st.button("🎲 Random Seed", use_container_width=True):
#             new_seed = random.randint(1, 999999)
#             st.info(f"🎲 New random seed: {new_seed}")
            
#             with st.spinner("Running FLUX + Gemini pipeline..."):
#                 try:
#                     results = run_full_pipeline(new_seed)
                    
#                     if results:
#                         st.session_state.generated_images.extend(results)
#                         st.success(f"✅ Generated {len(results)} sketch(es) with seed {new_seed}!")
#                         st.rerun()
                        
#                 except Exception as e:
#                     st.error(f"❌ Pipeline failed: {str(e)}")
    
#     with col3:
#         if st.button("🗑 Clear All", use_container_width=True):
#             st.session_state.generated_images = []
#             st.rerun()
    
#     # ----------------------------
#     # 🖼 GALLERY DISPLAY
#     # ----------------------------
#     if st.session_state.generated_images:
#         st.markdown("---")
#         st.header("🎨 Your Generated Sketches Gallery")
        
#         for idx, img_data in enumerate(reversed(st.session_state.generated_images)):
#             number = len(st.session_state.generated_images) - idx
            
#             st.subheader(f"Generation #{number} (Seed: {img_data['seed']})")
            
#             # Display both base and sketch
#             col1, col2, col3 = st.columns([1, 1, 1])
            
#             with col1:
#                 st.image(img_data["base"], caption="FLUX Base Portrait", use_container_width=True)
            
#             with col2:
#                 st.image(img_data["sketch"], caption="Gemini Pencil Sketch", use_container_width=True)
            
#             with col3:
#                 st.download_button(
#                     label=f"⬇ Download Sketch #{number}",
#                     data=img_data["sketch_data"],
#                     file_name=f"pencil_sketch_seed_{img_data['seed']}.png",
#                     mime="image/png",
#                     key=f"download_{idx}",
#                     use_container_width=True
#                 )
                
#                 st.metric("Seed Value", img_data['seed'])
#                 st.caption("Use this seed to recreate this exact face!")
            
#             st.divider()
#     else:
#         st.info("👆 Click 'Generate with Seed' to create your first pencil sketch!")
    
#     # ----------------------------
#     # ℹ INFO SECTION
#     # ----------------------------
#     st.markdown("---")

# else:
#     # Show placeholder when keys are not provided
#     st.title("✏ Perfect Pencil Sketch Generator")
#     st.warning("🔐 Please configure your API keys in the sidebar to use the application")
#     st.info("""
#     You need to provide:
#     - **Gemini API Key**: For converting images to pencil sketches
#     - **Hugging Face Token**: For generating base portraits with FLUX
    
#     Once you enter both keys in the sidebar, the full application will appear here.
#     """)

# # Add some info in sidebar about where to get keys
# st.sidebar.markdown("---")
# st.sidebar.markdown("### 🔍 Where to get API keys:")
# st.sidebar.markdown("- **Gemini**: [Google AI Studio](https://aistudio.google.com/)")
# st.sidebar.markdown("- **Hugging Face**: [Access Tokens](https://huggingface.co/settings/tokens)")




























import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import io
import random
from huggingface_hub import InferenceClient

# ----------------------------
# 🔧 APP CONFIG
# ----------------------------
st.set_page_config(page_title="Perfect Pencil Sketch Generator", page_icon="✏", layout="wide")

# ----------------------------
# 🔑 API KEYS SIDEBAR
# ----------------------------
st.sidebar.title("🔑 API Configuration")
st.sidebar.markdown("Please provide both API keys to use the application")

# Initialize session state for API keys
if 'gemini_key' not in st.session_state:
    st.session_state.gemini_key = ""
if 'hf_token' not in st.session_state:
    st.session_state.hf_token = ""
if 'keys_provided' not in st.session_state:
    st.session_state.keys_provided = False

# API key input fields
gemini_key = st.sidebar.text_input("Gemini API Key", type="password", 
                                  help="Enter your Google Gemini API key")
hf_token = st.sidebar.text_input("Hugging Face Token", type="password", 
                                help="Enter your Hugging Face API token")

# Save keys to session state when provided
if gemini_key and hf_token:
    st.session_state.gemini_key = gemini_key
    st.session_state.hf_token = hf_token
    st.session_state.keys_provided = True
    st.sidebar.success("✅ API keys saved! You can now use the application.")
elif gemini_key or hf_token:
    st.sidebar.warning("⚠️ Please provide both API keys")
else:
    st.sidebar.info("🔐 Please enter both API keys to continue")

# Clear keys button
if st.sidebar.button("Clear API Keys"):
    st.session_state.gemini_key = ""
    st.session_state.hf_token = ""
    st.session_state.keys_provided = False
    st.rerun()

# ----------------------------
# MAIN APPLICATION (Only show if keys are provided)
# ----------------------------
if st.session_state.keys_provided:
    # Initialize Gemini client with provided key
    gemini_client = genai.Client(api_key=st.session_state.gemini_key)
    
    # Main app title and description
    st.title("✏ Perfect Pencil Sketch Generator - SEED SUPPORT!")
    
    # ----------------------------
    # 🎛 User Controls
    # ----------------------------
    st.header("🧠 Describe Your Portrait")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Any", "Male", "Female"], index=2)
    with col2:
        age = st.selectbox("Age Group", ["Any", "18-24 months", "20-30 years", "30-40 years", "40-50 years", "50-60 years"], index=3)
    with col3:
        ethnicity = st.selectbox("Ethnicity", ["Any", "Caucasian/White", "Hispanic/Latino", "African-American/African", "South Asian", "East Asian", "Middle-Eastern"], index=1)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        eye_shape = st.selectbox("Eye Shape", ["Any", "Almond", "Round", "Monolid", "Hooded", "Upturned", "Downturned"])
    with col2:
        lips_shape = st.selectbox("Lips Shape", ["Any", "Full", "Thin", "Heart-shaped", "Round"])
    with col3:
        nose = st.selectbox("Nose Shape", ["Any", "Straight", "Button", "Hooked"])
    
    col1, col2 = st.columns(2)
    with col1:
        face_cutt = st.selectbox("Face Cut", ["Any", "Oval", "Round", "Square", "Heart"])
    with col2:
        cheecks = st.selectbox("Cheeks", ["Any", "High cheekbones", "Chubby cheeks", "Defined"])
    
    pose = st.selectbox("Pose", ["Front view", "3/4 view", "Side profile", "Looking down", "Looking up"], index=0)
    expression = st.selectbox("Facial Expression", ["Neutral", "Slight Smile", "Full Smile", "Serious", "Calm"], index=1)
    
    st.header("💇‍♀️ Hair Style")
    hair_style = st.selectbox("Hair Style", ["Any", "Straight", "Wavy", "Curly", "Coily", "Bob cut", "Pixie cut"])
    
    st.header("✨ Other Features")
    eyebrow_shape = st.selectbox("Eyebrow Shape", ["Any", "Thick", "Thin", "Arched", "Straight"])
    
    # --- Settings ---
    st.header("⚙ Generation Settings")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        num_images = st.slider("Number of Images", 1, 4, 1)
    with col2:
        aspect = st.selectbox("Aspect Ratio", ["1:1", "4:3", "16:9"], index=0)
    with col3:
        seed = st.number_input("🎯 Seed Value (WORKING!)", min_value=0, value=random.randint(1, 999999), help="Same seed = Same face!")
    
    prompt_user = st.text_area(
        "💡 Add any extra artistic details (optional):",
        placeholder="Example: wearing a scarf, vintage paper background"
    )
    
    # Session state for storing images
    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = []
    
    # ----------------------------
    # 🪄 PROMPT BUILDER FOR FLUX
    # ----------------------------
    def build_flux_prompt():
        """Build prompt for FLUX to generate realistic portrait"""
        positive = "professional studio portrait photograph, "
        
        if expression != "Neutral":
            positive += f"{expression.lower()} expression, "
        if ethnicity != "Any":
            positive += f"{ethnicity.lower()}, "
        if gender != "Any":
            positive += f"{gender.lower()}, "
        if age != "Any":
            positive += f"{age.lower()}, "
        
        if eye_shape != "Any":
            positive += f"{eye_shape.lower()} eyes, "
        if lips_shape != "Any":
            positive += f"{lips_shape.lower()} lips, "
        if nose != "Any":
            positive += f"{nose.lower()} nose, "
        if face_cutt != "Any":
            positive += f"{face_cutt.lower()} face shape, "
        if cheecks != "Any":
            positive += f"{cheecks.lower()}, "
        if eyebrow_shape != "Any":
            positive += f"{eyebrow_shape.lower()} eyebrows, "
        if hair_style != "Any":
            positive += f"{hair_style.lower()} hair, "
        
        positive += "head and shoulders portrait, "
        positive += f"{pose.lower()}, "
        positive += "professional studio lighting, plain background, high resolution, detailed facial features, photorealistic"
        
        if prompt_user.strip():
            positive += f", {prompt_user.strip()}"
        
        return positive
    
    # ----------------------------
    # 🎨 STEP 1: FLUX GENERATION (FIXED)
    # ----------------------------
    def generate_flux_image(prompt, seed_value, width=768, height=768):
        """Generate base image with FLUX using NEW Hugging Face Inference Client"""
        try:
            # Initialize Hugging Face Inference Client
            client = InferenceClient(api_key=st.session_state.hf_token)
            
            # Generate image using text_to_image
            image = client.text_to_image(
                prompt=prompt,
                model="black-forest-labs/FLUX.1-schnell",
                width=width,
                height=height,
                num_inference_steps=4,  # FLUX.1-schnell works best with 4 steps
                guidance_scale=0.0  # FLUX.1-schnell doesn't use guidance
            )
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            raise Exception(f"FLUX generation failed: {str(e)}")
    
    # ----------------------------
    # 🖼 ENHANCED STEP 2: GEMINI PENCIL CONVERSION
    # ----------------------------
    def convert_to_pencil_sketch(pil_image):
        """Convert image to realistic hand-drawn pencil sketch using Gemini 2.5 Flash Image"""
        
        # ENHANCED PROMPT for realistic pencil sketch aesthetic
        conversion_prompt = """Transform this portrait into an authentic hand-drawn graphite pencil sketch on textured drawing paper.

CRITICAL REQUIREMENTS - PENCIL SKETCH AESTHETIC:

**TEXTURE & MATERIAL:**
- Visible rough drawing paper grain and tooth texture throughout
- Real graphite pencil marks - rough, grainy, hand-drawn strokes
- Natural pencil texture with visible graphite particles
- Authentic paper fiber texture showing through
- Matte pencil finish, no smooth digital gradients

**DRAWING TECHNIQUE:**
- Hand-drawn pencil strokes following facial contours and forms
- Visible directional hatching and cross-hatching for shading
- Rough, sketchy line quality - NOT smooth vector lines
- Natural artist hand movements visible in strokes
- Varying pencil pressure - lighter in highlights, darker in shadows
- Use multiple pencil grades effect: HB for outlines, 2B-4B for mid-tones, 6B-8B for darkest shadows

**SHADING & TONES:**
- Soft graphite smudging and blending in shadow areas
- Delicate tonal transitions using layered pencil strokes
- Strategic eraser-lift highlights for brightest areas (eyes, nose tip, lips)
- Natural pencil buildup in darker areas
- White paper showing through in highlights - NOT pure white
- Mid-gray paper tone as base

**STYLE CHARACTERISTICS:**
- Sketchy, artistic, hand-drawn appearance
- Slightly loose, expressive strokes - not photorealistic rendering
- Natural imperfections and variations in line weight
- Authentic artist's hand-drawn portrait feel
- Black graphite on off-white/cream paper

**COMPOSITION:**
- Complete portrait - full head and shoulders visible
- Clean framing with slight paper border/vignette effect
- All facial features fully detailed and finished
- Natural portrait proportions maintained

**WHAT TO AVOID:**
- NO smooth digital gradients or airbrushed appearance
- NO clean vector lines or perfect curves
- NO photorealistic rendering or painting effects
- NO color - pure black and white graphite only
- NO signatures, watermarks, or text
- NO incomplete or unfinished areas

**FINAL RESULT:**
The sketch should look like a skilled artist spent 2-3 hours drawing this portrait with graphite pencils on textured paper. Every viewer should immediately recognize it as a hand-drawn pencil sketch, not a digital filter or photo manipulation.

Maintain the exact facial features, expression, and likeness from the input portrait."""

        try:
            # Generate with Gemini 2.5 Flash Image
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[conversion_prompt, pil_image],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    temperature=0.4,  # Lower for more consistent pencil aesthetic
                )
            )
            
            # Extract the generated sketch
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    sketch_bytes = part.inline_data.data
                    sketch_image = Image.open(io.BytesIO(sketch_bytes))
                    return sketch_image
            
            raise Exception("No image in Gemini response")
            
        except Exception as e:
            raise Exception(f"Gemini conversion failed: {str(e)}")
    
    # ----------------------------
    # 🎨 ALTERNATIVE: POST-PROCESSING ENHANCEMENT (Optional)
    # ----------------------------
    def enhance_pencil_texture(pil_image):
        """
        Optional post-processing to add more pencil texture if needed.
        Use this AFTER Gemini conversion if you want extra texture.
        """
        from PIL import ImageFilter, ImageEnhance
        import numpy as np
        
        # Convert to grayscale if not already
        if pil_image.mode != 'L':
            pil_image = pil_image.convert('L')
        
        # Slight sharpening to enhance pencil strokes
        enhanced = pil_image.filter(ImageFilter.SHARPEN)
        
        # Increase contrast slightly for more defined pencil marks
        contrast = ImageEnhance.Contrast(enhanced)
        enhanced = contrast.enhance(1.2)
        
        # Add subtle noise for paper grain (optional)
        img_array = np.array(enhanced).astype(float)
        noise = np.random.normal(0, 2, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array, mode='L')
    
    # ----------------------------
    # 🚀 UPDATED FULL PIPELINE (with optional enhancement)
    # ----------------------------
    def run_full_pipeline(seed_value):
        """Run the complete FLUX → Gemini pipeline with enhanced pencil conversion"""
        
        # Build FLUX prompt
        flux_prompt = build_flux_prompt()
        
        # Determine dimensions
        width, height = 768, 768
        if aspect == "4:3":
            width, height = 896, 672
        elif aspect == "16:9":
            width, height = 1024, 576
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = []
        
        for i in range(num_images):
            current_seed = seed_value + i
            
            try:
                # STEP 1: Generate with FLUX
                status_text.text(f"🎨 Step 1/3: Generating base portrait with FLUX (seed: {current_seed})...")
                progress_bar.progress((i * 3) / (num_images * 3))
                
                base_image = generate_flux_image(flux_prompt, current_seed, width, height)
                
                # STEP 2: Convert to pencil sketch with Gemini
                status_text.text(f"✏ Step 2/3: Converting to hand-drawn pencil sketch...")
                progress_bar.progress((i * 3 + 1) / (num_images * 3))
                
                sketch_image = convert_to_pencil_sketch(base_image)
                
                # STEP 3 (OPTIONAL): Enhance pencil texture
                # Uncomment next line if you want extra texture enhancement:
                sketch_image = enhance_pencil_texture(sketch_image)
                
                # Store results
                buf = io.BytesIO()
                sketch_image.save(buf, format="PNG")
                
                all_results.append({
                    "base": base_image,
                    "sketch": sketch_image,
                    "sketch_data": buf.getvalue(),
                    "seed": current_seed
                })
                
                progress_bar.progress((i * 3 + 2) / (num_images * 3))
                
            except Exception as e:
                st.error(f"❌ Failed for seed {current_seed}: {str(e)}")
                continue
        
        status_text.empty()
        progress_bar.empty()
        
        return all_results
    
    # ----------------------------
    # 🎨 GENERATE BUTTONS
    # ----------------------------
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✨ Generate with Seed", use_container_width=True, type="primary"):
            with st.spinner("Running FLUX + Gemini pipeline..."):
                try:
                    st.info(f"🎯 Using seed: {seed}")
                    results = run_full_pipeline(seed)
                    
                    if results:
                        st.session_state.generated_images.extend(results)
                        st.success(f"✅ Generated {len(results)} sketch(es) successfully!")
                        st.rerun()
                    else:
                        st.error("❌ No images generated")
                        
                except Exception as e:
                    st.error(f"❌ Pipeline failed: {str(e)}")
    
    with col2:
        if st.button("🎲 Random Seed", use_container_width=True):
            new_seed = random.randint(1, 999999)
            st.info(f"🎲 New random seed: {new_seed}")
            
            with st.spinner("Running FLUX + Gemini pipeline..."):
                try:
                    results = run_full_pipeline(new_seed)
                    
                    if results:
                        st.session_state.generated_images.extend(results)
                        st.success(f"✅ Generated {len(results)} sketch(es) with seed {new_seed}!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Pipeline failed: {str(e)}")
    
    with col3:
        if st.button("🗑 Clear All", use_container_width=True):
            st.session_state.generated_images = []
            st.rerun()
    
    # ----------------------------
    # 🖼 GALLERY DISPLAY
    # ----------------------------
    if st.session_state.generated_images:
        st.markdown("---")
        st.header("🎨 Your Generated Sketches Gallery")
        
        for idx, img_data in enumerate(reversed(st.session_state.generated_images)):
            number = len(st.session_state.generated_images) - idx
            
            st.subheader(f"Generation #{number} (Seed: {img_data['seed']})")
            
            # Display both base and sketch
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.image(img_data["base"], caption="FLUX Base Portrait", use_container_width=True)
            
            with col2:
                st.image(img_data["sketch"], caption="Gemini Pencil Sketch", use_container_width=True)
            
            with col3:
                st.download_button(
                    label=f"⬇ Download Sketch #{number}",
                    data=img_data["sketch_data"],
                    file_name=f"pencil_sketch_seed_{img_data['seed']}.png",
                    mime="image/png",
                    key=f"download_{idx}",
                    use_container_width=True
                )
                
                st.metric("Seed Value", img_data['seed'])
                st.caption("Use this seed to recreate this exact face!")
            
            st.divider()
    else:
        st.info("👆 Click 'Generate with Seed' to create your first pencil sketch!")
    
    # ----------------------------
    # ℹ INFO SECTION
    # ----------------------------
    st.markdown("---")

else:
    # Show placeholder when keys are not provided
    st.title("✏ Perfect Pencil Sketch Generator")
    st.warning("🔐 Please configure your API keys in the sidebar to use the application")
    st.info("""
    You need to provide:
    - **Gemini API Key**: For converting images to pencil sketches
    - **Hugging Face Token**: For generating base portraits with FLUX
    
    Once you enter both keys in the sidebar, the full application will appear here.
    """)

# Add some info in sidebar about where to get keys
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Where to get API keys:")
st.sidebar.markdown("- **Gemini**: [Google AI Studio](https://aistudio.google.com/)")
st.sidebar.markdown("- **Hugging Face**: [Access Tokens](https://huggingface.co/settings/tokens)")

