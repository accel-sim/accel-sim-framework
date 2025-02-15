#!/usr/bin/env python3

from optparse import OptionParser
import re
import os
import subprocess
from subprocess import Popen, STDOUT
import sys

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
data_root = os.path.join(this_directory, "..", "..", "hw_run")

cards_availible = [
    "fermi-gtx480",
    "kepler-titan",
    "pascal-titanx",
    "pascal-p100",
    "pascal-1080ti",
    "volta-titanv",
    "volta-quadro-v100",
    "volta-tesla-v100",
    "turing-rtx2060",
    "ampere-rtx3070",
]

parser = OptionParser()
parser.add_option(
    "-c",
    "--cards",
    dest="cards",
    default="all",
    help="Comma seperated list of cards to download. Available cards: {0}".format(
        cards_availible
    ),
)
(options, args) = parser.parse_args()

os.makedirs(data_root, exist_ok=True)

if "all" == options.cards:
    options.cards = ",".join(cards_availible)

for card in options.cards.split(","):
    if card not in cards_availible:
        sys.exit(
            'Error, wrong card name "{0}". Valid names: all or {1}'.format(
                card, cards_availible
            )
        )
    tarfilename = card + ".tgz"
    folder_path = os.path.join(data_root, card)
    if not os.path.exists(folder_path):
        subprocess.run(
            [
                "wget",
                "https://engineering.purdue.edu/tgrogers/gpgpu-sim/hw_data/{0}".format(
                    tarfilename
                ),
            ]
        )
        subprocess.run(["tar", "-xzvf", tarfilename, "-C", data_root])
        os.remove(tarfilename)
    else:
        print("Found " + folder_path)
