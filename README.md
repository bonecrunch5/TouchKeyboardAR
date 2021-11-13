# FEUP-RVA
## Getting Started
Install OpenCV and other dependencies on Python:

```
pip install opencv-python==4.5.4.58 python-dotenv
```

Create a `.env` file with the same structure as `.envsample` before running the program. Change the values for your specific environment.

## Running

The `preparation.py` program is used to get the necessary data from a keyboard to be used later. It receives an optional argument which is a plain text file with the ordered list of symbols on the keyboard (from left to right, up to down). By providing this file, the `preparation.py` will output every file needed for the next program in the `generated` folder. If you don't want to provide that file, check [**Creating the keys list file manually**](##Creating-the-keys-list-file-manually).

```
python preparation.py <key-symbol-list-file-path>
```

The `recognition.py` identifies the keyboard through the camera and detects user input following the user's finger. It receives two optional arguments: the path of the keyboard image and the path of the keys information file (which arre both outputted by the `preparation.py` program). If these arguments are not provided, the program will look for the files in their default paths (`generated/imgKeyboard.jpg` and `generated/keys.json` respectively).

```
python recognition.py <keyboard-image-path> <keys-list-file-path>
```

## Creating the keys list file manually

If for some reason you don't want the `preparation.py` program to generate the keys information file automatically, you can write it yourself. All you have to do is follow the JSON structure defined bellow:
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