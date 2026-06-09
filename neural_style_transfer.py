import os
import sys
import time
import requests
from PIL import Image
from io import BytesIO

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    col = True
except ImportError:
    col = False


def pr(text, color="white", bold=False):
    if not col:
        print(text)
        return
    c = {
        "cyan": Fore.CYAN, "green": Fore.GREEN,
        "yellow": Fore.YELLOW, "red": Fore.RED,
        "blue": Fore.LIGHTBLUE_EX, "white": Fore.WHITE,
        "magenta": Fore.MAGENTA
    }.get(color, Fore.WHITE)
    b = Style.BRIGHT if bold else ""
    print(f"{b}{c}{text}{Style.RESET_ALL}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 256 if torch.cuda.is_available() else 200

loader = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

unloader = transforms.ToPILImage()


def load_image(path_or_url):
    try:
        if path_or_url.startswith("http"):
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(path_or_url, headers=headers, timeout=15)
            img = Image.open(BytesIO(r.content)).convert("RGB")
        else:
            img = Image.open(path_or_url).convert("RGB")
        img = loader(img).unsqueeze(0)
        return img.to(device, torch.float)
    except Exception as e:
        pr(f"Could not load image: {e}", "red")
        return None


def save_image(tensor, filename):
    img = tensor.cpu().clone().squeeze(0)
    img = unloader(img)
    img.save(filename)
    return img


class GramMatrix(nn.Module):
    def forward(self, x):
        b, c, h, w = x.size()
        f = x.view(b * c, h * w)
        G = torch.mm(f, f.t())
        return G.div(b * c * h * w)


class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.loss = 0

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x


class StyleLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.gram = GramMatrix()
        self.target = self.gram(target).detach()
        self.loss = 0

    def forward(self, x):
        G = self.gram(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x


def build_model(content_img, style_img):
    pr("Loading VGG19 model...", "blue")
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    content_losses = []
    style_losses = []

    model = nn.Sequential()
    gram = GramMatrix()

    conv_count = 0
    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            conv_count += 1
            name = f"conv_{conv_count}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{conv_count}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{conv_count}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{conv_count}"
        else:
            name = f"layer_{conv_count}"

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            cl = ContentLoss(target)
            model.add_module(f"content_loss_{conv_count}", cl)
            content_losses.append(cl)

        if name in style_layers:
            target = model(style_img).detach()
            sl = StyleLoss(target)
            model.add_module(f"style_loss_{conv_count}", sl)
            style_losses.append(sl)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
    model = model[:i + 1]

    return model, content_losses, style_losses


def run_style_transfer(content_img, style_img, steps=300, style_weight=1000000, content_weight=1):
    pr("Building model...", "blue")
    model, content_losses, style_losses = build_model(content_img, style_img)

    input_img = content_img.clone()
    input_img.requires_grad_(True)

    optimizer = optim.LBFGS([input_img])

    pr(f"Running {steps} steps... please wait", "cyan", bold=True)
    pr("This takes 2-5 mins on CPU. Don't close the window!\n", "yellow")

    run = [0]
    start = time.time()

    while run[0] <= steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses) * style_weight
            content_score = sum(cl.loss for cl in content_losses) * content_weight
            loss = style_score + content_score
            loss.backward()
            run[0] += 1
            if run[0] % 50 == 0:
                elapsed = int(time.time() - start)
                pr(f"  Step {run[0]}/{steps} done | {elapsed}s elapsed", "white")
            return loss
        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


def create_sample_images():
    pr("Creating sample images locally...", "blue")

    # create a simple colorful content image
    content = Image.new("RGB", (200, 200))
    pixels = content.load()
    for i in range(200):
        for j in range(200):
            pixels[i, j] = (i, j, (i + j) % 255)
    content.save("sample_content.png")

    # create a simple style image (warm sunset colors)
    style = Image.new("RGB", (200, 200))
    pixels2 = style.load()
    for i in range(200):
        for j in range(200):
            pixels2[i, j] = (255, int(i * 0.8), int(j * 0.3))
    style.save("sample_style.png")

    pr("Sample images created: sample_content.png and sample_style.png", "green")
    return "sample_content.png", "sample_style.png"


def show_header():
    print()
    pr("=" * 60, "cyan", bold=True)
    pr("       NEURAL STYLE TRANSFER", "cyan", bold=True)
    pr("       Codetech Internship - Task 3", "cyan")
    pr("=" * 60, "cyan", bold=True)
    pr(f"  Device: {'GPU (CUDA)' if device.type == 'cuda' else 'CPU'}", "blue")
    pr(f"  Image Size: {img_size}x{img_size}", "blue")
    print()


def run():
    show_header()

    pr("Choose content image (your photo):", "yellow", bold=True)
    pr("  [1] Auto-generate a sample image (no internet needed)", "white")
    pr("  [2] Use your own image file (jpg/png)\n", "white")

    c1 = input("Enter choice (1/2): ").strip()

    if c1 == "1":
        content_path, auto_style = create_sample_images()
        pr(f"Using: {content_path}", "green")
    else:
        content_path = input("\n  Full path to your image: ").strip().strip('"')
        auto_style = None

    print()
    pr("Choose style image (the artwork):", "yellow", bold=True)
    pr("  [1] Use the auto-generated style", "white")
    pr("  [2] Use your own style image (jpg/png)\n", "white")

    c2 = input("Enter choice (1/2): ").strip()

    if c2 == "1":
        if auto_style:
            style_path = auto_style
        else:
            _, style_path = create_sample_images()
        pr(f"Using: {style_path}", "green")
    else:
        style_path = input("\n  Full path to style image: ").strip().strip('"')

    print()
    pr("How many steps?", "yellow", bold=True)
    pr("  [1] Fast   - 100 steps  (~1 min)", "white")
    pr("  [2] Normal - 300 steps  (~3 min)", "white")
    pr("  [3] Best   - 500 steps  (~6 min)\n", "white")

    sc = input("Enter choice (1/2/3): ").strip()
    steps = {"1": 100, "2": 300, "3": 500}.get(sc, 100)

    print()
    pr("Loading images...", "blue")
    content_img = load_image(content_path)
    style_img = load_image(style_path)

    if content_img is None or style_img is None:
        pr("Failed to load one or both images.", "red")
        return

    pr("Images loaded!", "green")
    print()

    output_tensor = run_style_transfer(content_img, style_img, steps=steps)

    output_file = "styled_output.png"
    save_image(output_tensor, output_file)

    print()
    pr("=" * 60, "cyan", bold=True)
    pr("  Style transfer complete!", "green", bold=True)
    pr(f"  Output saved as: {output_file}", "green")
    pr("  Open styled_output.png to see your result!", "green")
    pr("=" * 60, "cyan", bold=True)
    print()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)
