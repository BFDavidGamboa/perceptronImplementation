import numpy as np
# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    v1 = np.array([1, 2, 3])
    v2 = 0.5 * v1

    # Re-read this
    print(np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))


# Press the green button in the gutter to run the script.

main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
