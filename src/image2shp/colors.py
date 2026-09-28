
import numpy as np

def preds2colors(pred_mask):

    image = np.argmax(pred_mask, axis=0)
    #plt.imshow(image)
    #plt.savefig(str(output_path).replace(".png", "_2.png"))


    # Step 2: define 6 class colors (BGR)
    colors = np.array([
        [0, 0, 0],        # class 0 - black
        [255, 0, 0],      # class 1 - blue
        [0, 255, 0],      # class 2 - green
        [0, 0, 255],      # class 3 - red
        [255, 255, 0],    # class 4 - cyan
        [255, 0, 255],    # class 5 - magenta
    ], dtype=np.uint8)

    # Step 3: map labels to colors
    colored = colors[image]
    return colored
