import streamlit as st
from PIL import Image
from function import (encrypt_image, compress_image, generate_secret_key)
import os

download_ready = False

# Center align the title using HTML/CSS and make "Sender" a lighter yellow
st.markdown("""
    <h1 style='text-align: center;'>
        ETC <span style='color: #FFFF99;'>Sender</span> Application
    </h1>
    """,
            unsafe_allow_html=True)

# Load and display the local image
image = Image.open("Sender Image.png")
st.image(image, use_column_width=True)

# Add description of the application
st.write("""
    This application allows users to upload a TIF/TIFF image and download the processed TXT file. 
    You simply need to upload your image, and the application will extract the relevant text data from the image file, 
    which can then be downloaded as a text file for further use.

    **We recommend that TIFF files should not exceed 50MB.**  
    Larger files may take a significant amount of time to encrypt, and if a TIFF file exceeds 50MB, 
    the compressed text file could become extremely large, making it difficult for the receiver to decompress 
    and attach the file.
    """)

# Create columns for buttons
col1, col2, col3 = st.columns(3)

# Button for downloading the sample dataset
with col1:
    if st.button("Download Sample Dataset"):
        st.markdown(
            "[Click here to download the sample dataset](https://drive.google.com/drive/folders/115FN6YJLhMjPrILJbL7XKhfoonbABLAz?usp=share_link)"
        )

# Button for JPEG to TIFF converter
with col2:
    if st.button("Go to JPEG to TIFF Converter"):
        st.markdown(
            "[Click here to convert JPEG to TIFF](https://cloudconvert.com/jpeg-to-tiff)"
        )

# Button for resizing TIFF images
with col3:
    if st.button("  Go to TIFF Resizer"):
        st.markdown(
            "[Click here to resize TIFF images](https://www.xconvert.com/resize-tiff)"
        )

download_ready = False

compressed_file_data = None

with st.form("my_form", clear_on_submit=True):
    uploaded_files = st.file_uploader("Upload a TIF image",
                                      type=["tif", "tiff"])

    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB in bytes
    BLOCK_SIZE = 8
    SECRET_KEY = generate_secret_key()[1]
    submitted = st.form_submit_button("Submit")

    if submitted:
        if uploaded_files is None:
            st.error("No file uploaded. Please upload a TIF file.")
        else:
            # Check the file size
            if uploaded_files.size > MAX_FILE_SIZE:
                st.error(
                    f"The uploaded file exceeds the 50MB size limit. Please upload a smaller file."
                )
            else:
                # Proceed with image processing if file size is within the limit
                original_size = uploaded_files.size  # Original file size in bytes
                image = Image.open(uploaded_files)

            # Check if the image dimensions are divisible by BLOCK_SIZE
            if image.width % BLOCK_SIZE != 0 or image.height % BLOCK_SIZE != 0:
                st.error(
                    f"The dimensions of this image ({image.width}x{image.height}) are not properly adjusted. "
                    f"Please resize the image to dimensions divisible by 8 and try again."
                )

            else:
                scrambled_y_image, secret_key_str = encrypt_image(
                    image, BLOCK_SIZE, SECRET_KEY)

                # Show the encrypted image
                st.image(scrambled_y_image,
                         caption="Encrypted Image",
                         use_column_width=True)

                # Perform compression
                compressed_file_path = compress_image(scrambled_y_image, '.')

                # Get the size of the encrypted and compressed file
                encrypted_image_size = scrambled_y_image.size  # Size after encryption
                compressed_file_size = os.path.getsize(
                    compressed_file_path)  # Compressed file size in bytes

                # Display size information
                st.write(f"Original File Size: {original_size / 1024:.2f} KB")
                st.write(
                    f"Encrypted Image Size: {encrypted_image_size[0]}x{encrypted_image_size[1]}"
                )
                st.write(
                    f"Compressed File Size: {compressed_file_size / 1024:.2f} KB"
                )

                # Display the secret key centered
                st.markdown(
                    f"<div style='text-align: center; font-size: 20px;'>Secret Key: {secret_key_str}</div>",
                    unsafe_allow_html=True)

                # Display warning message
                st.warning(
                    "This Secret Key will only be displayed at the time of downloading the encrypted image. Please make sure to write it down somewhere else!"
                )

                with open(compressed_file_path, "rb") as file:
                    compressed_file_data = file.read()

                download_ready = True  # Set download readiness status

if download_ready and compressed_file_data is not None:
    # Display download button only when the file is ready
    download_button_clicked = st.download_button(
        label="Download Encrypted Data",
        data=compressed_file_data,
        file_name="compressed_data.txt",
        mime="text/plain")

    if download_button_clicked:  # When the download button is clicked
        st.success("Download was successful!")
