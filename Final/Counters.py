import cv2

def counter():
    input_img = cv2.imread("Pasel.jpg")
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(gray, 150, 1000, cv2.THRESH_BINARY)

    find_counters = dst.copy()

    couners_finded, herarhy = cv2.findContours(find_counters, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE);


    new_one = cv2.drawContours(input_img, couners_finded, -1, (255, 0, 0))

    while True:
        cv2.imshow('img', new_one)
        if cv2.waitKey(1) == 27:
            break


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    counter()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
