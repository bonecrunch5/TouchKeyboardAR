# FEUP-RVA
## Getting Started
Install OpenCV and other dependencies on Python:

```
pip install opencv-python==4.5.4.58 python-dotenv
```

Create a `.env` file with the same structure as `.envsample` before running the program. Change the values for your specific environment.

## Running

### Preparation Program

The **Preparation Program** is used to get the necessary data from a keyboard to be used later. It receives an argument which is a plain text file with the ordered list of symbols on the keyboard (from left to right, up to down). `preparation.py` will output every file needed for the next program in the `generated` folder. 

```
python preparation.py <key-symbol-list-file-path>
```

Follow the instructions on screen. Once the keyboard and keys have been detected, press <kbd>Space</kbd> to save the output. If you wish to quit without saving, press <kbd>Esc</kbd>.

If you don't want to run this program before running the **Recognition Program**, you'll need to create the needed files yourself. The first file is a top view image of the keyboard (the keyboard show fill the whole image). The second file is a keys information file; for that check [**Creating the keys list file manually**](#Creating-the-keys-list-file-manually).

### Recognition Program

The **Recognition Program** identifies the keyboard through the camera and detects user input following the user's finger. It receives two optional arguments: the path of the keyboard image and the path of the keys information file (which are both outputted by the **Preparation Program**). If these arguments are not provided, the program will look for the files in their default paths (`generated/imgKeyboard.jpg` and `generated/keys.json` respectively).

```
python recognition.py <keyboard-image-path> <keys-list-file-path>
```

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