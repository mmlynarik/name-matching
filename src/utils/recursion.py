import time


class StandardBacktrackingSplitter:

    def __init__(self, word_lengths=(2, 3)):
        self.word_lengths = word_lengths
        self.result = set()

    def split(self, text):
        self.result.clear()
        self.__try_level(text, "")

        if len(text) <= 0:
            return

        to_return = list(self.result)
        to_return.sort()

        return to_return

    # execute backtracking for each word length
    def __try_level(self, text, previous_word):
        result = False

        for length in self.word_lengths:
            partial_result = self.__try_level_with_length(text, length, previous_word)
            if partial_result:
                result = True

        return result

    # try given word
    def __try_level_with_length(self, text, length, previous_word):
        text_length = len(text)

        # word too long
        if text_length < length:
            return False

        # new word is same sa previous
        new_word = text[0:length]
        if new_word == previous_word:
            return False

        # new word fills the rest of the string, we have a match
        if text_length == length:
            self.result.add(new_word)
            return True

        # we have to go deeper
        result = self.__try_level(text[length:], new_word)

        # let's mark path as present
        if result:
            self.result.add(new_word)

        return result


class DynamicSplitter:

    def __init__(self, word_lengths=(2, 3)):
        self.word_lengths = word_lengths
        self.result = set()
        self.cache = dict()
        for i in word_lengths:
            self.cache[i] = dict()

    def split(self, text):
        self.result.clear()
        self.__try_level(text, 0, "")

        if len(text) <= 0:
            return

        to_return = list(self.result)
        to_return.sort()

        return to_return

    # execute backtracking for each word length
    def __try_level(self, text, path_length, previous_word):
        result = False

        for length in self.word_lengths:
            partial_result = self.__try_level_with_length(text, path_length, length, previous_word)
            if partial_result:
                result = True

        return result

    # try given word
    def __try_level_with_length(self, text, path_length, length, previous_word):
        text_length = len(text)

        # word too long
        if text_length < length:
            return False

        # new word is same sa previous
        new_word = text[0:length]
        if new_word == previous_word:
            return False

        # cached result for the result of the string exists, we don't need to go deeper to include
        # this combination too
        already_cached = self.cache[length].get(text_length)
        if already_cached:
            self.add_to_result(new_word)
            return True
        if not already_cached and already_cached is not None:
            return False

        # new word fills the rest of the string, we have a match
        if text_length == length:
            self.add_to_result_and_cache(text_length, length, new_word)
            return True

        # we have to go deeper
        result = self.__try_level(text[length:], path_length + length, new_word)

        # let's mark path as present
        if result:
            self.add_to_result_and_cache(text_length, length, new_word)
        else:
            self.add_to_cache(text_length, length, False)

        return result

    def add_to_result_and_cache(self, text_length, length, new_word):
        self.add_to_result(new_word)
        self.add_to_cache(text_length, length, True)

    def add_to_cache(self, text_length, length, valid):
        self.cache[length][text_length] = valid

    def add_to_result(self, new_word):
        self.result.add(new_word)


if __name__ == '__main__':
    print(StandardBacktrackingSplitter().split("ababab"))
    print(DynamicSplitter().split("ababab"))
    print(StandardBacktrackingSplitter().split("abcdef"))
    print(DynamicSplitter().split("abcdef"))
    print(StandardBacktrackingSplitter().split("abcdefg"))
    print(DynamicSplitter().split("abcdefg"))
    print(StandardBacktrackingSplitter().split("ababcabc"))
    print(DynamicSplitter().split("ababcabc"))
    print(StandardBacktrackingSplitter().split("ababcabc"))
    print(DynamicSplitter().split("ababcabc"))
    print(StandardBacktrackingSplitter().split("ccccacccc"))
    print(DynamicSplitter().split("ccccacccc"))

    time1 = time.time()
    print(StandardBacktrackingSplitter().split(
        "abcdefghijklmnopqrstuvwxyz123456789acbdefghijklmnop")
    )
    time2 = time.time()
    print(time2 - time1)
    print(DynamicSplitter().split("abcdefghijklmnopqrstuvwxyz123456789acbdefghijklmnop"))
    time3 = time.time()
    print(time3 - time2)
