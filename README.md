# FEUP-RVA
## Getting Started
Install OpenCV and other dependencies on Python:

```
pip install opencv-python==4.5.4.58 python-dotenv
```

### Environment Variables

Create a `.env` file with the same structure as `.envsample` before running the program and change the values for your specific environment. There are the following environment variables:

| **Variable**       | **Description**                                                                 | **Default Value** |
|--------------------|---------------------------------------------------------------------------------|-------------------|
| CAMERA_ID          | The ID of the webcam to be used in the program.                                 | 0                 |
| KEY_PRESS_DURATION | Amount of seconds with fingertip over key for it to be pressed.                 | 1                 |
| SHOW_DEBUG_IMAGES  | When `TRUE`, shows images of the various image processing steps of the program. | FALSE             |
| SHOW_KEY_CORNERS   | When `TRUE`, shows key corners on top view images of the keyboard.              | FALSE             |
| SHOW_KEY_LABELS    | When `TRUE`, shows key labels on top view images of the keyboard.               | FALSE             |
| SHOW_KEY_EDGES     | When `TRUE`, shows key edges on top view images of the keyboard.                | FALSE             |

The last 3 variables only affect the **Recognition Program**. The **Preparation Program** always shows key labels and edges for the user to verify if the scan is correct.

## Running

### Preparation Program

The **Preparation Program** is used to get the necessary data from a keyboard to be used later. It receives an argument which is a plain text file with the ordered list of symbols on the keyboard (from left to right, up to down). `preparation.py` will output every file needed for the next program in the `generated` folder. 

```
python preparation.py <key-symbol-list-file-path>
```

Follow the instructions on screen. Once the keyboard and keys have been detected, press <kbd>Space</kbd> to save the output. If you wish to quit without saving, press <kbd>Esc</kbd>.

If you don't want to run this program before running the **Recognition Program**, you'll need to create the needed files yourself. The first file is a top view image of the keyboard (the keyboard should fill the whole image). The second file is a keys information file; for that check [**Creating the keys list file manually**](#Creating-the-keys-list-file-manually).

### Recognition Program

The **Recognition Program** identifies the keyboard through the camera and detects user input following the user's finger. It receives two optional arguments: the path of the keyboard image and the path of the keys information file (which are both outputted by the **Preparation Program**). If these arguments are not provided, the program will look for the files in their default paths (`generated/imgKeyboard.jpg` and `generated/keys.json` respectively).

```
python recognition.py <keyboard-image-path> <keys-list-file-path>
```

Follow the instructions on screen (use <kbd>Space</kbd> to complete each instruction). Once you've done the initial steps (*hand scan* and *clear background scan* steps), you can use your hand to press the keys of the keyboard. Just hold your finger tip over the key you want to press, with your finger fully extended, for 1 second (you can change this value with the `KEY_PRESS_DURATION` environment variable).

The program will show the pressed keys on the screen, as well as the inputted text on the console terminal. When you exit the program, by pressing <kbd>Esc</kbd>, the inputted text will also be written to a file in the `kb-output` folder. 

It should work unregardless of the keyboard and finger orientation relative to the camera. If it is behaving badly, you can redo both the *hand scan* step and *clear background scan* step by pressing <kbd>H</kbd> or <kbd>B</kbd> respectively. If it continues to behave badly, your lighting setup may not be appropriate.

## Creating the keys list file manually

If for some reason you don't want to run **Preparation Program**, you can write the keys information file yourself. All you have to do is follow the JSON structure defined bellow:
```
{
    "keys": [
        {
            "points": [
                {
                    "x": <value>,
                    "y": <value>
                },
                ... (3 more)
            ],
            "symbol": <value>
        },
        ...
    ]
}
```