# Emotion Detection and Filter Application

## Authors

- Dong Nguyen
- Parth Gupta
- Ka Hin Choi

## Purpose

This program uses OpenCV to detect human faces, eyes, and smiles on a video stream. It determines human emotions (happy or sad) based on collected facial features and applies filters accordingly. When no face is detected or the local computer camera is unavailable, it shows appropriate messages. It adds a teardrop filter below the human eye when detecting a sad emotion and a rainbow on the mouth when detecting a happy emotion.

## Installation

1. Clone and open the repository:

   ```sh
   git clone https://github.com/parthgu/EmotifilterFX
   ```

2. Create and activate a virtual environment:

    - **Create virtual environment:**
      ```sh
      python -m venv myenv
      ```

    - **Activate the virtual environment:**
      - **Mac/Linux:**
        ```sh
        source myenv/bin/activate
        ```
      - **Windows:**
        ```sh
        myenv\Scripts\activate
        ```

4. Install the required dependencies:

   ```sh
   pip install -r requirements.txt
   ```

5. Ensure you have the necessary cascade files:

   - `haarcascade_frontalface_default.xml`
   - `haarcascade_smile.xml`
   - `haarcascade_eye.xml`

   If you don't have them, you can download them from the [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades) and place them in a directory named `cascades`.

6. Ensure you have the filter images:
   - `teardrop.png`
   - `rainbowsmile.png`
   - `angry-eyebrows,png`
   - `fire.png`
   - `exclamation.png`

## Usage

1. Activate the virtual environment if not already activated:

   - **Mac/Linux:**
     ```sh
     source myenv/bin/activate
     ```
   - **Windows:**
     ```sh
     myenv\Scripts\activate
     ```

2. Run the program:

   ```sh
   python main.py
   ```

3. The program will open a window displaying the video stream from your camera. If your computer has multiple cameras, you might need to edit `main.py` and change the value of `video = cv2.VideoCapture(1)` to `video = cv2.VideoCapture(0)` or another index to select the correct camera.

4. The program will detect faces, eyes, and smiles, and apply the corresponding filters based on detected emotions:

   - A teardrop filter will be applied below the eye when no smile is detected (indicating sadness).
   - A rainbow filter will be applied on the mouth when a smile is detected (indicating happiness).

5. Press the `Esc` key (ASCII 27) to close the video window and stop the program.

## Troubleshooting

- If the program cannot open the camera, you will see an error message: "Error: Could not open camera."
- If the program cannot load the face, smile, or eye detectors, you will see error messages indicating which detector could not be loaded.
- Ensure that the paths to the cascade files and filter images are correct and the files exist in the specified directories.

## License

This project is licensed under the MIT License.
