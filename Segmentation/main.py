from tkinter.filedialog import askopenfilename
from obj.gui import draw_gui
import sys

from tracker import stats_from_param

if "-stats" in sys.argv:
    stats_from_param()
else:
    directory = askopenfilename(initialdir="/")
    print(directory)
    gui = draw_gui(directory)
