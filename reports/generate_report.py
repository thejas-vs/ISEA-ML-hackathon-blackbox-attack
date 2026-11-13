import os
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from PIL import Image

proj_root = r'C:/Users/theja/Downloads/ISEA ML/blackbox-attack'
report_path = os.path.join(proj_root, 'reports', 'report.pdf')
fig_dir = os.path.join(proj_root, 'reports', 'figures')

c = canvas.Canvas(report_path, pagesize=LETTER)
width, height = LETTER
margin = 0.5 * inch
x = margin
y = height - margin

def draw_heading(text, size=16, space=0.2*inch):
    global y
    c.setFont('Helvetica-Bold', size)
    c.drawString(x, y, text)
    y -= size + space

def draw_paragraph(text, size=10, space=0.15*inch):
    global y
    c.setFont('Helvetica', size)
    lines = text.split('\n')
    line_h = size + 2
    for line in lines:
        c.drawString(x, y, line)
        y -= line_h
    y -= space

def add_image(img_name, caption, max_width=6*inch):
    global y
    img_path = os.path.join(fig_dir, img_name)
    if not os.path.isfile(img_path):
        draw_paragraph(f'[Missing image: {img_name}]')
        return
    im = Image.open(img_path)
    iw, ih = im.size
    aspect = ih / iw
    draw_width = min(max_width, iw * 0.75)
    draw_height = draw_width * aspect
    if y - draw_height < margin:
        c.showPage()
        y = height - margin
    c.drawImage(img_path, x, y - draw_height, width=draw_width, height=draw_height)
    y -= draw_height + 0.1*inch
    c.setFont('Helvetica-Oblique', 9)
    c.drawString(x, y, caption)
    y -= 0.2*inch

# Title
c.setFont('Helvetica-Bold', 20)
c.drawCentredString(width/2, y, 'Black‑Box Decision‑Based Adversarial Attack')
y -= 30
c.setFont('Helvetica', 12)
c.drawCentredString(width/2, y, 'Repository: blackbox-attack')
y -= 40

# Sections
draw_heading('Problem Statement', 14)
draw_paragraph('Generate adversarial PNG images for a pretrained CNN (models/cnn_model.pt) using only label‑only (decision‑based) queries. The dataset contains 1 000 clean PNGs in data/clean/. The goal is to produce visually indistinguishable adversarial images that cause a mis‑classification.')

draw_heading('Black‑Box Constraints', 14)
draw_paragraph('- No access to model internals – only BlackBoxModel.predict_label.\n- No internet or package installation during execution.\n- Deterministic seed = 42.\n- No pushes to remote repositories.')

draw_heading('Method Overview (HopSkipJumpAttack style)', 14)
method_text = (
    '1. Initialization – generate a random‑noise image (or blend with a random image) that is already mis‑classified.\n'
    '2. Binary search to the decision boundary – shrink the line segment between the clean image and the adversarial point until we land just on the boundary (tolerance 1e‑5).\n'
    '3. Gradient‑direction estimation – sample random directions on the unit sphere, query the model for their labels, and estimate a gradient sign via Monte‑Carlo.\n'
    '4. Geometric step – move opposite to the estimated gradient; if the step crosses back to the clean side we halve the step size and retry, then binary‑search again toward the clean image.\n'
    '5. Iteration – repeat steps 3‑4 for a configurable number of iterations or until the per‑image query budget is exhausted. Monte‑Carlo sample count grows with iteration number.\n'
    '6. Safety check – after the loop, round the image to uint8 and clip to [0,255]; if rounding flips the label back, move a tiny additional distance away from the clean image until the label stays adversarial.\n'
    '7. Optional refinement – scalar shrinkage, sparsification, and low‑pass smoothing using only label queries.'
)
draw_paragraph(method_text)

draw_heading('Final Hyper‑Parameters & Seed', 14)
hyper = (
    '- seed: 42\n'
    '- budget per image: 3000 label queries\n'
    '- max iterations: 8\n'
    '- max_num_evals (initial MC samples): 20\n'
    '- tolerance (binary‑search): 1e‑5\n'
    '- refinement enabled: True\n'
    '- shrink tolerance: 1e‑4\n'
    '- sparsify block size: 4 × 4\n'
    '- smooth sigma: 0.7'
)
draw_paragraph(hyper)

draw_heading('Results Summary (1000 images)', 14)
results = (
    'Success rate: 100 %\n'
    'Mean L₂: ≈ 0.96\n'
    'Mean L∞: ≈ 0.102\n'
    'Mean PSNR (dB): ≈ 44.4\n'
    'Mean SSIM: ≈ 0.9982\n'
    'Mean queries per image: ≈ 3 125'
)
draw_paragraph(results)

draw_heading('Key Figures', 14)
add_image('validation_example_1.png', 'Figure 1: Clean | Adversarial | Amplified noise (example 1)')
add_image('validation_example_2.png', 'Figure 2: Clean | Adversarial | Amplified noise (example 2)')
add_image('hsja_curve_1.png', 'Figure 3: L₂ vs. Queries curve (subset)')

draw_heading('Limitations & Future Work', 14)
limit = (
    '- Query budget: ~3 k queries per image (~3 M total) may be costly on stricter query‑limited platforms.\n'
    '- Refinement provides modest L₂ improvements at the cost of extra queries; can be disabled for speed.\n'
    '- Results are tied to the specific CNN checkpoint; harder models may require larger budgets or more iterations.\n'
    '- Visible perturbations become noticeable under strong magnification despite high PSNR/SSIM.\n'
    '- Future work: explore adaptive query allocation, smarter initialization, or combine with gradient‑free optimization techniques.'
)
draw_paragraph(limit)

draw_heading('Reproduction Instructions', 14)
repro = (
    '1. Install the required packages (see requirements.txt).\n'
    '2. Place the model checkpoint cnn_model.pt in models/ (if not already present).\n'
    '3. Run the notebook: jupyter notebook code/p2_code.ipynb (executes a small demo on 3 images).\n'
    '4. Generate the full set of adversarial images:\n'
    '   python code/run_all.py --start 0 --end 1000\n'
    '   The script resumes automatically if interrupted.\n'
    '5. Validate the outputs (optional): python code/validate_outputs.py.'
)
draw_paragraph(repro)

c.save()
print('PDF created at', report_path)
