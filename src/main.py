# -*- coding: utf-8 -*-
import ahocorasick
import pickle
import gzip
import morfeusz2
import pandas as pd
from pyxdameraulevenshtein import damerau_levenshtein_distance


class TextCorrectorPL:
    def __init__(self):
        with gzip.open("static/words_trie.pkl.gz", "rb") as f:
            self.words_trie = pickle.load(f)

        self.morfeusz = morfeusz2.Morfeusz()

        self.qwerty_typos = {
            "q": ["a", "w"],
            "w": ["q", "a", "s", "e"],
            "e": ["w", "s", "d", "r"],
            "r": ["e", "d", "f", "t"],
            "t": ["r", "f", "g", "y"],
            "y": ["t", "g", "h", "u"],
            "u": ["y", "h", "j", "i"],
            "i": ["u", "j", "k", "o"],
            "o": ["i", "k", "l", "p"],
            "p": ["o", "l"],
            "a": ["q", "w", "s", "z"],
            "s": ["a", "w", "e", "d", "x", "z"],
            "d": ["s", "e", "r", "f", "c", "x"],
            "f": ["d", "r", "t", "g", "v", "c"],
            "g": ["f", "t", "y", "h", "b", "v"],
            "h": ["g", "y", "u", "j", "n", "b"],
            "j": ["h", "u", "i", "k", "m", "n"],
            "k": ["j", "i", "o", "l", "m"],
            "l": ["k", "o", "p"],
            "z": ["a", "s", "x"],
            "x": ["z", "s", "d", "c"],
            "c": ["x", "d", "f", "v"],
            "v": ["c", "f", "g", "b"],
            "b": ["v", "g", "h", "n"],
            "n": ["b", "h", "j", "m"],
            "m": ["n", "j", "k"],
        }

    def ___is_in_morfeusz_dict(self, word):
        return self.morfeusz.analyse(word)[0][2][2] != "ign"

    def __find_freq(self, word):
        return self.words_trie.get(word, [0, ""])[0]

    def __construct_dataframe(self, candidates, weight):
        df_temp = pd.DataFrame(candidates)
        df_temp.columns = "word"
        df_temp["weight"] = weight
        return df_temp

    def __replace_with_diacritic(self, words: set, index: int) -> set:
        """return set of various word combination for possible diacritic letter variations"""

        diacritic = {
            "e": "ę",
            "o": "ó",
            "a": "ą",
            "s": "ś",
            "l": "ł",
            "c": "ć",
            "n": "ń",
        }

        output_words = set()
        output_words.update(words)
        letter = next(iter(words))[index]
        if letter == "z":
            for word in words:
                output_words.update([word[:index] + "ż" + word[index + 1 :]])
                output_words.update([word[:index] + "ź" + word[index + 1 :]])
        elif letter in diacritic.keys():
            for word in words:
                output_words.update(
                    [word[:index] + diacritic[letter] + word[index + 1 :]]
                )
        return output_words

    def __diacritic_combinations(self, word: str) -> list:
        """gather all word diacritic word variations and return list with no duplicates"""

        all_combinations = set([word])
        for index in range(len(word)):
            all_combinations.update(
                self.__replace_with_diacritic(all_combinations, index)
            )
        return list(all_combinations)

    def __remove_one_letter_from_word(self, word):
        candidates = list()
        for index in range(len(word)):
            candidates.append(word[:index] + word[index + 1 :])
        return candidates

    def __swap_two_adjacent_letters(self, word):
        candidates = list()
        for index in range(len(word) - 1):
            candidates.append(
                word[:index] + word[index + 1] + word[index] + word[index + 2 :]
            )
        return candidates

    def __qwerty_keyboard_typos(self, word):
        candidates = list()
        for index in range(len(word)):
            for typo in self.qwerty_typos[word[index]]:
                candidates.append(word[:index] + typo + word[index + 1 :])
        return candidates

    def __reduce_qwerty_keyboard_typo(self, word):
        candidates = list()
        for index in range(len(word) - 1):
            if word[index] in self.qwerty_typos[word[index + 1]]:
                candidates.append(word[:index] + word[index] + word[index + 2 :])
                candidates.append(word[:index] + word[index + 1] + word[index + 2 :])
        return candidates

    def __additional_wildcard(self, word):
        "no wildcard at the end as it may indicate word's plural form"
        candidates = list()
        for index in range(len(word) - 1):
            candidates.append(word[:index] + "?" + word[index:])
        return candidates

    def __replace_letter_with_wildcard(self, word):
        candidates = list()
        for index in range(len(word)):
            candidates.append(word[:index] + "?" + word[index:])
        return candidates
