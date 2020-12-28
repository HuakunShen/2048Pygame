def get_divider_str(msg: str, length: int = 100, line_ends: str = ''):
    result = ""
    space_left = length - 2 - len(msg)
    if space_left <= 0:
        return msg
    elif space_left % 2 == 0:
        left, right = space_left // 2, space_left // 2
    else:
        left, right = space_left // 2, space_left // 2 + 1
    return f"{line_ends}{left * '='} {msg} {right * '='}{line_ends}"


import random


class A:
    @staticmethod
    def get_rand_int(a=0, b=3):
        return A.get_rand_int_help(a, b)

    @staticmethod
    def get_rand_int_help(a=0, b=3):
        return random.randint(a, b)


if __name__ == '__main__':
    random.seed(2048)
    print(A.get_rand_int())
