from PIL import Image, ImageDraw, ImageFont

# Load the image
img_path = '/mnt/data/image.png'
image = Image.open(img_path)

# Define the text translations
translations = {
    "Time": "时间",
    "Feature Extractor": "特征提取器",
    "Wav2vec": "Wav2vec",
    "Duration": "持续时间",
    "Pronunciation": "发音",
    "Attention Mechanism": "注意力机制",
    "Back-end": "后端",
    "LCNN": "LCNN",
    "Bi-LSTM": "双向LSTM",
    "GAP": "全局平均池化",
    "FC": "全连接层",
    "fake": "假",
    "real": "真"
}

# Define the position of texts in the image (x, y, text)
positions = [
    (370, 17, "Time"),
    (45, 130, "Feature Extractor"),
    (70, 185, "Wav2vec"),
    (220, 185, "Duration"),
    (390, 185, "Pronunciation"),
    (205, 250, "Attention Mechanism"),
    (180, 385, "Back-end"),
    (80, 440, "LCNN"),
    (200, 440, "Bi-LSTM"),
    (330, 440, "GAP"),
    (55, 495, "FC"),
    (180, 495, "FC"),
    (40, 535, "fake"),
    (235, 535, "real")
]

# Define the font and size
# Note: You may need to adjust the font and size according to your specific needs
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font_size = 10
font = ImageFont.truetype(font_path, font_size)

# Draw the translated text on the image
draw = ImageDraw.Draw(image)
for pos in positions:
    # Translate the text
    text = translations[pos[2]]
    # Clear the area where the new text will be drawn
    text_width, text_height = draw.textsize(pos[2], font=font)
    draw.rectangle([pos[0], pos[1], pos[0] + text_width, pos[1] + text_height], fill='white')
    # Draw the new text
    draw.text((pos[0], pos[1]), text, fill='black', font=font)

# Save the modified image
modified_img_path = '/mnt/data/modified_image.png'
image.save(modified_img_path)