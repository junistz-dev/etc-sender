import random
import numpy as np
from PIL import Image
import heapq
import json
import base64
from collections import defaultdict, Counter
import secrets
import time
import os

def block_scramble(image_array, block_size, key):
    """Scrambles the blocks of the image based on the provided key."""
    height, width = image_array.shape
    blocks = []

    # Divide the image into blocks
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image_array[i:i + block_size, j:j + block_size]
            blocks.append(block)

    # Generate random indices for scrambling
    indices = list(range(len(blocks)))
    random.seed(key)
    random.shuffle(indices)

    # Scramble blocks according to the shuffled indices
    scrambled_blocks = [blocks[i] for i in indices]

    scrambled_array = np.zeros_like(image_array)
    idx = 0
    # Reassemble scrambled blocks into the image
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            scrambled_array[i:i + block_size,
                            j:j + block_size] = scrambled_blocks[idx]
            idx += 1

    return scrambled_array, indices


def block_rotation(block, secret_key):
    """Rotates the block by 90 degrees according to a random value based on the secret key."""
    random.seed(secret_key)
    operation = random.randint(0, 3)

    # Rotate the block 90, 180, or 270 degrees based on the random operation
    if operation == 0:
        return np.rot90(block, 1)
    elif operation == 1:
        return np.rot90(block, 2)
    elif operation == 2:
        return np.rot90(block, 3)
    else:
        return block


def block_inversion(block, secret_key):
    """Flips the block horizontally or vertically based on the secret key."""
    random.seed(secret_key)
    operation = random.randint(0, 1)

    # Flip the block horizontally or vertically
    if operation == 0:
        return np.fliplr(block)
    else:
        return np.flipud(block)


def negative_positive_transformation(block, secret_key):
    """Applies negative-positive transformation to the block based on the secret key."""
    random.seed(secret_key)
    operation = random.randint(0, 1)

    # Apply negative transformation or keep the block unchanged
    if operation == 0:
        return 255 - block
    return block


def block_descramble(scrambled_array, indices, block_size):
    """Descrambles the blocks of the image based on the provided indices."""
    height, width = scrambled_array.shape

    # Divide scrambled image into blocks
    blocks = []
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = scrambled_array[i : i + block_size, j : j + block_size]
            blocks.append(block)

    # Reassemble blocks based on the original indices
    original_blocks = [None] * len(blocks)
    for i, idx in enumerate(indices):
        original_blocks[idx] = blocks[i]

    descrambled_array = np.zeros_like(scrambled_array)
    idx = 0
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            descrambled_array[i : i + block_size, j : j + block_size] = original_blocks[idx]
            idx += 1

    return descrambled_array


def block_derotation(block, secret_key):
    """Reverses the rotation of the block based on the secret key."""
    random.seed(secret_key)
    operation = random.randint(0, 3)

    # Reverse the rotation applied during encryption
    if operation == 0:
        return np.rot90(block, 3)
    elif operation == 1:
        return np.rot90(block, 2)
    elif operation == 2:
        return np.rot90(block, 1)
    else:
        return block


def block_deinversion(block, secret_key):
    """Reverses the inversion of the block based on the secret key."""
    random.seed(secret_key)
    operation = random.randint(0, 1)

    # Reverse the flip applied during encryption
    if operation == 0:
        return np.fliplr(block)
    else:
        return np.flipud(block)


def de_negative_positive_transformation(block, secret_key):
    """Reverses the negative-positive transformation based on the secret key."""
    random.seed(secret_key)
    operation = random.randint(0, 1)

    # Reverse the negative-positive transformation
    if operation == 0:
        return 255 - block
    return block


# Node class to represent the tree structure for Huffman coding
class Node:

    def __init__(self, symbol, frequency):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency


# Function to build the Huffman Tree
def build_huffman_tree(frequencies):
    priority_queue = [
        Node(symbol, freq) for symbol, freq in frequencies.items()
    ]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        merged = Node(None, left.frequency + right.frequency)
        merged.left = left
        merged.right = right
        heapq.heappush(priority_queue, merged)

    return priority_queue[0]


# Function to generate Huffman codes from the Huffman Tree
def generate_huffman_codes(root, current_code, codes):
    if root is None:
        return

    if root.symbol is not None:
        codes[root.symbol] = current_code
        return

    generate_huffman_codes(root.left, current_code + "0", codes)
    generate_huffman_codes(root.right, current_code + "1", codes)


def generate_secret_key():
    secret_key = [secrets.randbits(16) for _ in range(4)]
    secret_key_str = "-".join(map(str, secret_key))
    return secret_key, secret_key_str


def compress_channel(channel):
    # Flatten the channel into a 1D array
    pixels = channel.flatten()

    # Calculate the frequency of each pixel value
    frequencies = dict(Counter(pixels))

    # Build the Huffman Tree and generate Huffman codes
    huffman_tree = build_huffman_tree(frequencies)
    codes = {}
    generate_huffman_codes(huffman_tree, "", codes)

    # Encode the channel using the Huffman codes
    encoded_channel = ''.join([codes[pixel] for pixel in pixels])
    return encoded_channel, huffman_tree


def decompress_channel(encoded_channel, huffman_tree, shape):
    decoded_pixels = []
    current_node = huffman_tree

    for bit in encoded_channel:

        # Traverse the Huffman Tree based on the encoded bits
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right

        # If a leaf node is reached, append the corresponding pixel value
        if current_node.symbol is not None:
            decoded_pixels.append(current_node.symbol)
            current_node = huffman_tree

    # Reshape the decoded pixels into the original channel shape
    decoded_channel = np.array(decoded_pixels, dtype=np.uint8).reshape(shape)
    return decoded_channel


def encrypt_image(image,
                  block_size: int,
                  secret_key: str | None = None):
    # Convert the image to grayscale (2)
    grayscale_image = image.convert("L")

    # Split RGB to YCbCr (2.1)
    ycbcr_image = grayscale_image.convert("YCbCr")
    y, cb, cr = ycbcr_image.split()

    # Divide the image into blocks 8x8 (2.2)
    y_array = np.array(y)

    # Generate secret key (2.3)
    if secret_key is not None:
        secret_key_str = secret_key
        secret_key = [int(k) for k in secret_key.split("-")]
    else:
        secret_key, secret_key_str = generate_secret_key()
    """ Encryption """
    # Scramble the blocks of the image (2.4)
    scrambled_y_array, _ = block_scramble(y_array, block_size, secret_key[0])

    height, width = scrambled_y_array.shape

    # Rotate, invert, and transform the blocks (2.5)
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = scrambled_y_array[i:i + block_size, j:j + block_size]
            rotated_block = block_rotation(block, secret_key[1])
            inverted_block = block_inversion(rotated_block, secret_key[2])
            transformed_block = negative_positive_transformation(
                inverted_block, secret_key[3])
            scrambled_y_array[i:i + block_size,
                              j:j + block_size] = transformed_block

    # Convert the scrambled Y channel back to an image
    return Image.fromarray(scrambled_y_array), secret_key_str


def compress_image(image, output_path):
    y_array, cb_array, cr_array = np.array(image.convert("YCbCr").split())
    compressed_y, huffman_tree_y = compress_channel(y_array)
    compressed_data = {
        'y': compressed_y,
        'huffman_tree_y': serialize_tree(huffman_tree_y),
        'shape': y_array.shape,
    }
    compressed_file_path = os.path.join(output_path, "compressed_data.txt")
    with open(compressed_file_path, "w") as file:
        json.dump(compressed_data, file)
    return compressed_file_path


def serialize_tree(node: Node):

    if node.symbol is not None:
        return {
            'symbol': int(node.symbol) if node.symbol is not None else None,
            'frequency': node.frequency
        }
    return {
        'frequency': node.frequency,
        'left': serialize_tree(node.left),
        'right': serialize_tree(node.right)
    }


def deserialize_tree(data):
    if 'symbol' in data:
        return Node(data['symbol'], data['frequency'])
    node = Node(None, data['frequency'])
    node.left = deserialize_tree(data['left'])
    node.right = deserialize_tree(data['right'])
    return node


def decompress_image(file):
    data = json.load(file)

    encoded_y = data['y']
    encoded_shape = data['shape']
    huffman_tree_y = deserialize_tree(data['huffman_tree_y'])

    y_array = decompress_channel(encoded_y, huffman_tree_y, encoded_shape)

    decompressed_image = Image.fromarray(y_array)

    return decompressed_image


def decrypt_image(image, secret_key_str, block_size = 8):

    # Convert the image to YCbCr and extract the Y channel
    y_array, cb, cr = np.array(image.convert("YCbCr").split())

    # Parse the secret key
    parsed_secret_key = [int(k) for k in secret_key_str.split("-")]

    height, width = y_array.shape

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = y_array[i : i + block_size, j : j + block_size]
            transformed_block = de_negative_positive_transformation(
                block, parsed_secret_key[3]
            )
            inverted_block = block_deinversion(transformed_block, parsed_secret_key[2])
            rotated_block = block_derotation(inverted_block, parsed_secret_key[1])
            y_array[i : i + block_size, j : j + block_size] = rotated_block


    # Generate random indices for descrambling
    indices = list(range((height // block_size) * (width // block_size)))
    random.seed(parsed_secret_key[0])
    random.shuffle(indices)

    descrambled_y_array = block_descramble(y_array, indices, block_size)

    y_image = Image.fromarray(descrambled_y_array)

    return y_image


def jpeg_compress(
    image_path,
    output,
    quality=75,
):
    with Image.open(image_path) as img:
        img.save(f"{output}/compressed_image.jpg", "JPEG", quality=quality)

    return f"{output}/compressed_image.jpg"


def jpeg_decompress(image_path, output):
    output_path = f"{output}/receiver_decompressed_jpeg_image.tiff"
    with Image.open(f"{image_path}/compressed_image.jpg") as img:
        img.save(output_path, "TIFF")

    decompressed_image = Image.open(output_path)

    return decompressed_image, output_path


def png_compress(image_path, output):
    with Image.open(image_path) as img:
        img.save(f"{output}/compressed_image.png", format="PNG")

    return f"{output}/compressed_image.png"


def png_decompress(image_path, output):
    output_path = f"{output}/receiver_decompressed_png_image.tiff"
    with Image.open(f"{image_path}/compressed_image.png") as img:
        img.save(output_path, "TIFF")

    decompressed_image = Image.open(output_path)

    return decompressed_image, output_path


def compare_cte(data_path, secret_key, block_size, output_dir,
                original_image_path):
    decompression_start_time = time.time()
    decompressed_image = decompress_image(output_dir, output_dir)
    decompressed_image.save(output_dir + "/receiver_decompressed_image.tiff")
    decompression_time = time.time() - decompression_start_time

    jpeg_decompression_start_time = time.time()
    jpeg_decompressed_image = jpeg_decompress(
        output_dir + "/compressed_image.jpg",
        output_dir + "/receiver_decompressed_jpeg_image.tiff")
    jpeg_decompression_time = time.time() - jpeg_decompression_start_time

    png_decompression_start_time = time.time()
    png_decompressed_image = png_decompress(
        output_dir + "/compressed_image.png",
        output_dir + "/receiver_decompressed_png_image.tiff")
    png_decompression_time = time.time() - png_decompression_start_time

    decryption_start_time = time.time()
    huffman_original_image = decrypt_image(decompressed_image, secret_key,
                                           block_size)
    huffman_original_image.save(output_dir +
                                "/receiver_decompressed_decrypted_image.tiff")
    decryption_time = time.time() - decryption_start_time

    psnr = calculate_psnr(
        original_image_path,
        output_dir + "/receiver_decompressed_decrypted_image.tiff")
    ssim = calculate_ssim(
        original_image_path,
        output_dir + "/receiver_decompressed_decrypted_image.tiff")

    jpeg_decryption_start_time = time.time()
    jpeg_original_image = decrypt_image(jpeg_decompressed_image, secret_key,
                                        block_size)
    jpeg_original_image.save(
        output_dir + "/receiver_decompressed_decrypted_jpeg_image.tiff")
    jpeg_decryption_time = time.time() - jpeg_decryption_start_time

    png_decryption_start_time = time.time()
    png_original_image = decrypt_image(png_decompressed_image, secret_key,
                                       block_size)
    png_original_image.save(output_dir +
                            "/receiver_decompressed_decrypted_png_image.tiff")
    png_decryption_time = time.time() - png_decryption_start_time

    jpeg_psnr = calculate_psnr(
        original_image_path,
        output_dir + "/receiver_decompressed_decrypted_jpeg_image.tiff")
    jpeg_ssim = calculate_ssim(
        original_image_path,
        output_dir + "/receiver_decompressed_decrypted_jpeg_image.tiff")

    png_psnr = calculate_psnr(
        original_image_path,
        output_dir + "/receiver_decompressed_decrypted_png_image.tiff")
    png_ssim = calculate_ssim(
        original_image_path,
        output_dir + "/receiver_decompressed_decrypted_png_image.tiff")

    print(f"Decompression time: {decompression_time:.4f} seconds")
    print(f"Decryption time: {decryption_time:.4f} seconds")
    print(f"PSNR: {psnr:.4f} dB")
    print(f"SSIM: {ssim:.4f}")

    print(f"JPEG Decompression time: {jpeg_decompression_time:.4f} seconds")
    print(f"JPEG Decryption time: {jpeg_decryption_time:.4f} seconds")
    print(f"JPEG PSNR: {jpeg_psnr:.4f} dB")
    print(f"JPEG SSIM: {jpeg_ssim:.4f}")

    print(f"PNG Decompression time: {png_decompression_time:.4f} seconds")
    print(f"PNG Decryption time: {png_decryption_time:.4f} seconds")
    print(f"PNG PSNR: {png_psnr:.4f} dB")
    print(f"PNG SSIM: {png_ssim:.4f}")


def calculate_psnr(original_path, other_image_path):
    original_array = np.array(Image.open(original_path).convert("L"))
    other_array = np.array(Image.open(other_image_path).convert("L"))

    return peak_signal_noise_ratio(original_array, other_array)


def calculate_ssim(original_path, other_image_path):
    original_array = np.array(Image.open(original_path).convert("L"))
    other_array = np.array(Image.open(other_image_path).convert("L"))

    return structural_similarity(original_array,
                                 other_array,
                                 multichannel=True)
