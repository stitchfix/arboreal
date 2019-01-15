from sniffer.api import *
import os
import random
import subprocess
import sys
from collections import defaultdict


def speak_str(s):
    torun = 'say "'
    torun += s
    torun += '"'
    subprocess.call(torun, shell=True)


def speak_status(success=False):
    language = (
        "english"
    )  # 'english' or 'french', depending on OS voice settings used

    phrases = defaultdict(defaultdict)
    # let's put some english phrases in here
    phrases["english"]["pass"] = [
        "Joy to the world!",
        "Congratulations!",
        "Jolly good!",
        "Well done!",
        "Bravo!",
        "Three cheers!",
        "Hip Hip Horay!",
        "All's well that ends well",
    ]
    phrases["english"]["fail"] = [
        "Terribly sorry.",
        "Darn it.",
        "Oh, fiddlesticks.",
        "Back to work then!",
        "Better luck next time?",
        "What a shame.",
    ]
    phrases["english"]["pass_termination"] = " Tests pass."
    phrases["english"]["fail_termination"] = " Tests did not pass."
    # and some french phrases
    phrases["french"]["pass"] = [
        "Très bien fait!",
        "Félicitations!",
        "Bravo!",
        "Magnifique!",
    ]
    phrases["french"]["fail"] = [
        "Quelle dommage!",
        "Je suis navré!",
        "Zut alors!",
        "Ah mince!",
    ]
    phrases["french"]["pass_termination"] = " Tests passés."
    phrases["french"]["fail_termination"] = " Tests ratés."

    if success:
        message = (
            random.choice(phrases[language]["pass"])
            + phrases[language]["pass_termination"]
        )
    else:
        message = (
            random.choice(phrases[language]["fail"])
            + phrases[language]["fail_termination"]
        )

    speak_str(message)


# This makes @runnable fire automatically only if files ending with .py extension and not prefixed with a period are changed.
@file_validator
def py_files(filename):
    return filename.endswith(".py") and not os.path.basename(
        filename
    ).startswith(".")


# This example simply runs nose.
@runnable
def run_test_command(*args):
    import subprocess

    use_test_preset = (
        "all"
    )  # choices: demo, all, all_pdb, focus, focus_and_pdb
    test_presets = {
        "demo": "nosetests --nocapture --detailed-errors",
        "all": "nosetests --with-timer --timer-top-n=10 --detailed-errors --nocapture --verbose",
        "all_pdb": "nosetests --with-timer --timer-top-n=10 --detailed-errors --nocapture --verbose --pdb",
        "focus": "nosetests --with-timer --timer-top-n=10 --detailed-errors --nocapture --verbose --with-focus",
        "focus_pdb": "nosetests --with-timer --timer-top-n=10 --detailed-errors --nocapture --verbose --with-focus --pdb",
    }
    command = test_presets[use_test_preset]
    print(f"Running test command: {command}")
    rv = subprocess.call(command, shell=True)
    speak_status(success=(rv == 0))  # speak
    return rv == 0
