
import numpy as np
from skimage import io 
from PIL import Image, ImageDraw

# Create a circle
im = Image.new('L', (500, 300))
draw = ImageDraw.Draw(im) 
draw.ellipse((200, 100, 300, 200), 255)
im.save('circle.png', quality=95)
del draw, im 

# Create a square
im = Image.new('L', (500, 300))
draw = ImageDraw.Draw(im)
draw.rectangle((200, 100, 300, 200), 255)
im.save('rectangle.png', quality=95)
del draw, im

# Create a romb
im = Image.new('L', (500, 300))
draw = ImageDraw.Draw(im)
draw.polygon((200, 150, 250, 200, 300, 150), 255)
draw.polygon((200, 150, 250, 100, 300, 150), 255)
im.save('romb.png', quality=95)
del draw, im

# Create a ellipse
im = Image.new('L', (500, 300))
draw = ImageDraw.Draw(im) 
draw.ellipse((150, 100, 350, 200), 255)
im.save('ellipse.png', quality=95)
del draw, im 

# Create a triangle
im = Image.new('L', (500, 300))
draw = ImageDraw.Draw(im)
draw.polygon((150, 100, 250, 200, 350, 100), 255)
im.save('triangle.png', quality=95)
del draw, im

n = 100 # Create n noisy images
nclasses = 2
# images = [io.imread('ellipse.png',pilmode="L"), io.imread('rectangle.png',pilmode="L")]
images = [io.imread('ellipse.png',pilmode="L"), io.imread('circle.png',pilmode="L")]
# images = [io.imread('romb.png',pilmode="L"), io.imread('rectangle.png',pilmode="L")]
# images = [io.imread('triangle.png',pilmode="L"), io.imread('rectangle.png',pilmode="L")]
# images = [io.imread('romb.png',pilmode="L"), io.imread('triangle.png',pilmode="L")]

noisy_images = np.zeros((n,images[0].shape[0], images[0].shape[1]))
classes = np.zeros((n))

# add noise
for i in range(n):
  c = np.random.randint(0, nclasses)
  noisy_images[i] = np.random.poisson(images[c]*0.001)
  classes[i] = c
np.save('classes.npy',classes)

# Calculate cc matrix
cc_matrix = np.corrcoef(noisy_images.reshape(n,-1))

#save to file
file = open("cc_file.txt", "w")
for i in range(0, cc_matrix.shape[1]):
    for j in range(0, i+1):
        file = open("cc_file.txt", "a")
        file.write("%s % s %.2f" % (i+1, j+1, cc_matrix[i,j]) + '\n')
