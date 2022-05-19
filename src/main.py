# -*- coding: utf-8 -*-
import ahocorasick
import pickle
import gzip
import morfeusz2
import pandas as pd
import re
from string import punctuation
from pyxdameraulevenshtein import damerau_levenshtein_distance


class TextCorrectorPL:
    def __init__(self):
        with gzip.open("static/words_trie.pkl.gz", "rb") as f:
            self.__words_trie = pickle.load(f)

        self.__morfeusz = morfeusz2.Morfeusz()

        self.__qwerty_typos = {
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

    def ___is_in_morfeusz_dict(self, token: str) -> bool:
        """check if token exists in morfeusz dictionary"""
        return self.__morfeusz.analyse(token)[0][2][2] != "ign"

    def __find_freq(self, token: str) -> int:
        """get frequency of the word from words trie"""
        return self.__words_trie.get(token, [0, ""])[0]

    def __construct_dataframe(self, candidates, weight):
        df_temp = pd.DataFrame(candidates)
        df_temp.columns = "word"
        df_temp["weight"] = weight
        return df_temp

    def __replace_with_diacritic(self, tokens: set, index: int) -> set:
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
        output_words.update(tokens)
        letter = next(iter(tokens))[index]
        if letter == "z":
            for word in tokens:
                output_words.update([word[:index] + "ż" + word[index + 1 :]])
                output_words.update([word[:index] + "ź" + word[index + 1 :]])
        elif letter in diacritic.keys():
            for word in tokens:
                output_words.update(
                    [word[:index] + diacritic[letter] + word[index + 1 :]]
                )
        return output_words

    def __diacritic_combinations(self, token: str) -> list:
        """gather all word diacritic word variations and return list with no duplicates"""

        all_combinations = set([token])
        for index in range(len(token)):
            all_combinations.update(
                self.__replace_with_diacritic(all_combinations, index)
            )
        return list(all_combinations)

    def __remove_one_letter_from_word(self, token: str) -> list:
        """return list of original token but with removed every single letter"""
        candidates = list()
        for index in range(len(token)):
            candidates.append(token[:index] + token[index + 1 :])
        return candidates

    def __swap_two_adjacent_letters(self, token: str) -> list:
        """for given token swap every adjecent letter and return as a list"""
        candidates = list()
        for index in range(len(token) - 1):
            candidates.append(
                token[:index] + token[index + 1] + token[index] + token[index + 2 :]
            )
        return candidates

    def __qwerty_keyboard_typos(self, word):
        candidates = list()
        for index in range(len(word)):
            for typo in self.__qwerty_typos[word[index]]:
                candidates.append(word[:index] + typo + word[index + 1 :])
        return candidates

    def __reduce_qwerty_keyboard_typo(self, word):
        candidates = list()
        for index in range(len(word) - 1):
            if word[index] in self.__qwerty_typos[word[index + 1]]:
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

    def __check_if_token_is_upper(self, token):
        return token[0].upper()

    def __find_candicates(token):
        pass

    def correct_text(self, sequence):
        seq_split = [item for sublist in [elem.split() if not elem.isspace() else elem for elem in [re.sub(f'([{punctuation}0-9])', r' \1 ', elem) for elem in re.split(r'(\s+)', sequence)]] for item in sublist]
        # this list of comprehension above consists of following steps:
        # re.split(r'(\s+)', sequence) - split sequence but not by whitespaces (whitespaces are becoming elements of list)
        # re.sub(f'([{punctuation}0-9])', r' \1 ', elem) - add whitespaces before and after punctuation and numbers
        # iterate through elements in list and split them (separator is whitespace) if element is not space (.isspace() method)
        # flaten the list
        text_corrected = list()
        for token in seq_split:
            if token.isspace():
                text_corrected.append(token)
            elif token in punctuation:
                text_corrected.append(token)
            elif self.___is_in_morfeusz_dict(token):
                text_corrected.append(token)
            else:
                pass
        return "".join(text_corrected)

    