"""
Contact Decoder — a signal from the void.

Someone left a message buried in this number.
Run it. See what surfaces.
"""

# A single number. Harmless. Inert. Until you look closer.
# Every 8 bits is a character, stacked from left to right.
# Peel them off, one by one, and the message reveals itself.
PACKET = 1049828329182828377836188424028956231872836066810540958828448096995906530114164858833542306891413389834415981


def unpack(n):
    """Extract UTF-8 characters from a big-endian byte integer."""
    chars = []
    while n:
        chars.append(chr(n & 0xFF))
        n >>= 8
    return "".join(reversed(chars))


def reveal(text):
    """Print one character at a time. Suspense is free."""
    import sys
    import time

    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(0.025)


if __name__ == "__main__":
    import time

    print()
    print("   *    .  *       .   *    .  *       .   *    .  *")
    print(" .   *       .   *   .   *       .   *   .          *")
    print("   .     *    .     *    .     *    .     *    .     *")
    print(" *       .   *       .   *       .   *       .   *")
    print()
    print("   TRANSMISSION INBOUND...")
    print("   |")
    time.sleep(0.6)
    print("   |  Decrypting  [", end="", flush=True)
    for _ in range(20):
        time.sleep(0.04)
        print("#", end="", flush=True)
    print("] DONE")
    print("   |")
    time.sleep(0.3)
    print("   |  Unpacking...")
    time.sleep(0.3)
    print("   v")
    print()
    print("   " + "-" * 44)
    print()
    print("   ", end="")
    reveal(unpack(PACKET))
    print()
    print()
    print("   " + "-" * 44)
    print()
    print("   SIGNAL FADED. END OF TRANSMISSION.")
    print()
