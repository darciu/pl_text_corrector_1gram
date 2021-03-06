# -*- coding: utf-8 -*-
import re
import pickle
import gzip
from string import punctuation
import unicodedata
import itertools
from typing import Callable, List, Union

import pandas as pd
import morfeusz2
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
            "ę": ["w", "s", "d", "r", "ś"],
            "r": ["e", "d", "f", "t"],
            "t": ["r", "f", "g", "y"],
            "y": ["t", "g", "h", "u"],
            "u": ["y", "h", "j", "i"],
            "i": ["u", "j", "k", "o"],
            "o": ["i", "k", "l", "p"],
            "ó": ["i", "k", "l", "p", "ł"],
            "p": ["o", "l"],
            "a": ["q", "w", "s", "z"],
            "ą": ["q", "w", "s", "z", "ż", "ś"],
            "s": ["a", "w", "e", "d", "x", "z"],
            "ś": ["a", "w", "e", "d", "x", "z", "ą", "ę", "ż", "ź"],
            "d": ["s", "e", "r", "f", "c", "x"],
            "f": ["d", "r", "t", "g", "v", "c"],
            "g": ["f", "t", "y", "h", "b", "v"],
            "h": ["g", "y", "u", "j", "n", "b"],
            "j": ["h", "u", "i", "k", "m", "n"],
            "k": ["j", "i", "o", "l", "m"],
            "l": ["k", "o", "p"],
            "ł": ["k", "o", "p", "ł"],
            "z": ["a", "s", "x"],
            "ż": ["a", "s", "x", "ą", "ś", "ź"],
            "ź": ["a", "s", "x", "ą", "ś", "ż"],
            "x": ["z", "s", "d", "c"],
            "c": ["x", "d", "f", "v"],
            "ć": ["x", "d", "f", "v"],
            "v": ["c", "f", "g", "b"],
            "b": ["v", "g", "h", "n"],
            "n": ["b", "h", "j", "m"],
            "ń": ["b", "h", "j", "m"],
            "m": ["n", "j", "k"],
        }

    def ___is_in_morfeusz_dict(self, token: str) -> bool:
        """check if token exists in morfeusz dictionary"""
        return self.__morfeusz.analyse(token)[0][2][2] != "ign"

    def __produce_candidates(self, token_list: list, function: Callable) -> list:
        """helper function to loop for tokens in list"""
        token_set = set()
        for token in token_list:
            token_set.update(function(token))
        return list(token_set)

    def __remove_xd(self, token: str) -> str:
        """remove 'xd' phrase from the end of the token"""
        if token[-2:] == "xd":
            return token[:-2]
        return token

    def __candidates_reduce_repeated_letters(self, token: str) -> list:
        """reduce multiple char occurrence for the single one"""
        return list(set([re.sub(r"([a-z])\1+", r"\1", token), token]))

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

    def __candidates_diacritic_combinations(self, token: str) -> list:
        """gather all diacritic token variations and return list with no duplicates"""

        # # remove diacritics
        def strip_accents(s):
            return "".join(
                c
                for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn"
            ).replace("ł", "l")

        token = strip_accents(token)
        all_combinations = set([token])
        for index in range(len(token)):
            all_combinations.update(
                self.__replace_with_diacritic(all_combinations, index)
            )
        return list(all_combinations)

    def __replace_with_phonetic(self, tokens: set, index: int) -> set:
        """return set of various word combination for possible phonetic letter variations"""

        phonetic = {
            "k": "g",
            "g": "k",
            "p": "b",
            "b": "p",
            "d": "t",
            "t": "d",
            "w": "f",
            "f": "w",
            "z": "s",
            "s": "z",
        }

        output_words = set()
        output_words.update(tokens)
        letter = next(iter(tokens))[index]

        if letter in phonetic.keys():
            for word in tokens:
                output_words.update(
                    [word[:index] + phonetic[letter] + word[index + 1 :]]
                )
        return output_words

    def __candidates_phonetic_combinations(self, token: str) -> list:
        """gather all phonetic token variations and return list with no duplicates"""

        all_combinations = set([token])
        for index in range(len(token)):
            all_combinations.update(
                self.__replace_with_phonetic(all_combinations, index)
            )
        return list(all_combinations)

    def __all_elements_combinations(self, lst: list) -> List[List[tuple]]:
        """retrun all combinations of elements for given list"""
        return [list(itertools.combinations(lst, i)) for i in range(1, len(lst) + 1)]

    def __spelling_combinations(self, token: str, tup: tuple) -> list:
        """return combinations of given token according to provided spelling correction"""
        len_1 = len(tup[0])
        len_2 = len(tup[1])
        if len_1 > len_2:
            token = token.replace(tup[0], tup[0] + " " * (len_2 - len_1))
            tup = (tup[0] + " " * (len_2 - len_1), tup[1])
            len_1 = len(tup[0])
            len_2 = len(tup[1])
        token_combinations = list()
        for comb in self.__all_elements_combinations(
            [m.start() for m in re.finditer(tup[0], token)]
        ):
            for perm in comb:
                token_copy = token
                for pos in perm:
                    token_copy = (
                        token_copy[:pos]
                        + token_copy[pos : pos + len_1].replace(
                            tup[0], tup[1] + " " * (len_1 - len_2)
                        )
                        + token_copy[pos + len_1 :]
                    )
                token_combinations.append(token_copy.replace(" ", ""))
        return token_combinations

    def __candidates_common_spelling_errors(self, token: str) -> list:
        """return set of various word combination for most common spelling mistakes"""
        output_words = set()
        output_words.update([token])
        if token[-2:] == "jo":
            output_words.update([token[:-2] + "ją"])
        elif token[-1] == "i":
            output_words.update([token + "i"])

        for p in [
            ("en", "ę"),
            ("ua", "ła"),
            ("ue", "łe"),
            ("uy", "ły"),
            ("om", "ą"),
            ("ą", "om"),
            ("on", "ą"),
            ("rz", "sz"),
            ("sz", "rz"),
            ("rz", "ż"),
            ("ż", "rz"),
            ("ch", "h"),
            ("h", "ch"),
            ("u", "ó"),
            ("ó", "u"),
            ("cz", "trz"),
            ("dz", "c"),
        ]:
            temp_set = set()
            for word in output_words:
                temp_set.update(self.__spelling_combinations(word, p))
            output_words = output_words.union(temp_set)
        return output_words

    def __candidates_remove_one_letter(self, token: str) -> list:
        """return list of original token but with removed every single letter"""

        candidates = list()
        for index in range(len(token)):
            candidates.append(token[:index] + token[index + 1 :])
        return candidates

    def __candidates_swap_two_adjacent_letters(self, token: str) -> list:
        """for given token swap every adjecent letter and return as a list"""

        candidates = list()
        for index in range(len(token) - 1):
            candidates.append(
                token[:index] + token[index + 1] + token[index] + token[index + 2 :]
            )
        return candidates

    def __candidates_qwerty_keyboard_typos(self, token: str) -> list:
        """
        return list of all token variations but every letter is replaced with one of qwerty
        keyboard typos (look at __qwerty_typos dictionary)
        """

        candidates = list()
        for index in range(len(token)):
            for typo in self.__qwerty_typos.get(token[index], []):
                candidates.append(token[:index] + typo + token[index + 1 :])
        return candidates

    def __candidates_reduce_qwerty_keyboard_typo(self, token: str) -> list:
        """
        return list of all token variations but every possible qwerty keyboard double typo is reduced to one letter
        (eg. worlkd gives world and workd)
        """
        candidates = list()
        for index in range(len(token) - 1):
            if token[index] in self.__qwerty_typos.get(token[index + 1], []):
                candidates.append(token[:index] + token[index] + token[index + 2 :])
                candidates.append(token[:index] + token[index + 1] + token[index + 2 :])
        return candidates

    def __candidates_additional_wildcard(self, token: str) -> list:
        """
        return list of all token variations but every two adjacent letters are filled with wildcard in between
        no wildcard at the end as it may indicate word's plural form
        """
        candidates = list()
        for index in range(len(token) - 1):
            candidates.append(token[:index] + "?" + token[index:])
        return candidates

    def __candidates_replace_letter_with_wildcard(self, token: str) -> list:
        """
        return list of all token variations but every letter is replaced with wildcard

        """
        candidates = list()
        for index in range(len(token)):
            candidates.append(token[:index] + "?" + token[index:])
        return candidates

    def __check_if_token_is_capitalize(self, token: str) -> bool:
        return token[0].isupper()

    def __check_if_token_is_upper(self, token: str) -> bool:
        return token.isupper()

    def __split_sequence(self, sequence: str) -> list:
        """return splitted sequence but separate tokens are also punctuation, numbers and whitespaces"""
        # this list of comprehension above consists of following steps:
        # re.split(r'(\s+)', sequence) - split sequence but not by whitespaces (whitespaces are becoming elements of list)
        # re.sub(f'([{punctuation}0-9])', r' \1 ', elem) - add whitespaces before and after punctuation and numbers
        # iterate through elements in list and split them (separator is whitespace) if element is not space (.isspace() method)
        # flaten the list
        seq_split = [
            item
            for sublist in [
                elem.split() if not elem.isspace() else elem
                for elem in [
                    re.sub(f"([{punctuation}0-9])", r" \1 ", elem)
                    for elem in re.split(r"(\s+)", sequence)
                ]
            ]
            for item in sublist
        ]
        return seq_split

    def __find_freq(self, token: str) -> int:
        """get frequency of the word from words trie"""
        return self.__words_trie.get(token, [0, ""])[0]

    def __construct_dataframe(
        self, candidates: list, weight: float, candidate_type: str
    ) -> pd.DataFrame:
        """return pandas dataframe containing column with given candidates and weights"""
        df = pd.DataFrame(candidates)
        df.rename(columns={0: "token"}, inplace=True)
        df["weight"] = weight
        df["candidate_type"] = candidate_type
        return df

    def __wildcard_candidates_to_dataframe(
        self,
        df: pd.DataFrame,
        wildcard_candidates: list,
        weight: float,
        candidate_type: str,
    ) -> pd.DataFrame:
        df = df[["token", "weight", "freq", "candidate_type"]].copy()
        for candidate in wildcard_candidates:
            l = list(self.__words_trie.values(candidate, "?"))
            if len(l) > 0:
                for elem in l:

                    df.loc[len(df)] = [elem[1], weight, elem[0], candidate_type]

        return df

    def __create_df_with_candidates(self, token: str) -> pd.DataFrame:

        reduced_candidates = self.__candidates_reduce_repeated_letters(token)

        #   1st, most common step are diacritic or phonetic mistakes
        diacritic_candidates = self.__produce_candidates(
            reduced_candidates, self.__candidates_diacritic_combinations
        )
        diacritic_candidates.remove(token)

        phonetic_candidates = self.__produce_candidates(
            reduced_candidates, self.__candidates_phonetic_combinations
        )
        phonetic_candidates.remove(token)

        # most common grammar polish mistakes
        spelling_errors_candidates = self.__produce_candidates(
            reduced_candidates, self.__candidates_common_spelling_errors
        )

        spelling_errors_diacritic_candidates = self.__produce_candidates(
            spelling_errors_candidates, self.__candidates_diacritic_combinations
        )

        qwerty_typos_candidates = self.__produce_candidates(
            reduced_candidates, self.__candidates_qwerty_keyboard_typos
        )

        reduced_qwetry_typo_candidates = self.__produce_candidates(
            reduced_candidates, self.__candidates_reduce_qwerty_keyboard_typo
        )

        remove_one_letter_diacritic_candidates = self.__produce_candidates(
            reduced_candidates, self.__candidates_remove_one_letter
        )

        swap_two_adjacent_letters_candidates = self.__produce_candidates(
            reduced_candidates, self.__candidates_swap_two_adjacent_letters
        )

        additional_wildcard_candidates = self.__candidates_additional_wildcard(token)

        replaced_letter_with_wildcard_candidates = (
            self.__candidates_replace_letter_with_wildcard(token)
        )

        df = (
            self.__construct_dataframe(diacritic_candidates, 0.25, "diacritic")
            .append(self.__construct_dataframe(phonetic_candidates, 1.75, "phonetic"))
            .append(
                self.__construct_dataframe(
                    spelling_errors_candidates, 1.25, "spelling error"
                )
            )
            .append(
                self.__construct_dataframe(
                    spelling_errors_diacritic_candidates,
                    4.5,
                    "spelling error + diacritic",
                )
            )
            .append(
                self.__construct_dataframe(
                    qwerty_typos_candidates, 4.25, "qwerty typos"
                )
            )
            .append(
                self.__construct_dataframe(
                    reduced_qwetry_typo_candidates, 4.75, "reduced qwerty typos"
                )
            )
            .append(
                self.__construct_dataframe(
                    remove_one_letter_diacritic_candidates,
                    4.0,
                    "removed one letter + diacritic",
                )
            )
            .append(
                self.__construct_dataframe(
                    swap_two_adjacent_letters_candidates,
                    1.75,
                    "swapped adjacent letters",
                )
            )
        )

        df["freq"] = df["token"].apply(self.__find_freq)

        df = self.__wildcard_candidates_to_dataframe(
            df, additional_wildcard_candidates, 8.75, "additional wildcard"
        )
        df = self.__wildcard_candidates_to_dataframe(
            df,
            replaced_letter_with_wildcard_candidates,
            7.75,
            "letter replaced with wildcard",
        )

        df = df[df["freq"] > 0]

        return df

    def __correction_engine(self, token: str, analyze_token=False) -> Union[str, tuple]:
        """
        Main purpose of this part of the code is to collect as many resonable variations (candidates) for given token,
        find frequency of this variation in words_trie, calculate damerau-levenshtein distance from original token and variation
        and finally multiply everything including wages (there are different wages for different types of candicates).
        The top result with highest score will be returned as the most relevant token.
        """

        token = re.sub(r"[^ \na-zęóąśłżźćń/]+", "", token)
        token = self.__remove_xd(token)

        df = self.__create_df_with_candidates(token)

        if len(df) > 0:
            df["token"] = df["token"].astype(str)
            df["token"] = df["token"].fillna("")
            df["calibrated_distance"] = (
                df["token"].apply(lambda x: damerau_levenshtein_distance(str(x), token))
                + 1
            )
            df["result"] = df["freq"] / (df["weight"] + df["calibrated_distance"])
            df.sort_values(
                by="result", ascending=False, inplace=True, ignore_index=True
            )
            if analyze_token == False:
                return df["token"].iloc[0]
            else:
                return (df["token"].iloc[0], df["candidate_type"].iloc[0])
        else:
            if analyze_token == False:
                return token
            else:
                return (token, "not found candidate")

    def analyze_correct_text(self, sequence: str) -> pd.DataFrame:

        original_token = []

        corrected_token = []

        candidate_type = []

        seq_split = self.__split_sequence(sequence)

        for token in seq_split:
            original_token.append(token)
            if token.isspace():
                """token is whitespace"""
                corrected_token.append(token)
                candidate_type.append("whitespace")
            elif token in punctuation:
                """token is punctuation"""
                corrected_token.append(token)
                candidate_type.append("punctuation")
            elif token.isnumeric():
                """token is numeric"""
                corrected_token.append(token)
                candidate_type.append("numeric")
            elif len(token) <= 3:
                """token is shorter than three letters"""
                corrected_token.append(token)
                candidate_type.append("short token")
            elif self.___is_in_morfeusz_dict(token):
                """token already exists in polish dictionary"""
                corrected_token.append(token)
                candidate_type.append("in morfeusz dictionary")
            else:
                if self.__check_if_token_is_upper(token):
                    token_correct = self.__correction_engine(
                        token.lower(), analyze_token=True
                    ).upper()
                elif self.__check_if_token_is_capitalize(token):
                    token_correct = self.__correction_engine(
                        token.lower(), analyze_token=True
                    ).capitalize()
                else:
                    token_correct = self.__correction_engine(
                        token.lower(), analyze_token=True
                    )
                corrected_token.append(token_correct[0])
                candidate_type.append(token_correct[1])

            data = {
                "original_token": original_token,
                "corrected_token": corrected_token,
                "candidate_type": candidate_type,
            }

        return pd.DataFrame.from_dict(data)

    def correct_text(self, sequence: str) -> str:

        seq_split = self.__split_sequence(sequence)

        text_corrected = list()
        for token in seq_split:
            if len(token) <= 2:
                """token is shorter than three letters"""
                text_corrected.append(token)
            elif token.isspace():
                """token is whitespace"""
                text_corrected.append(token)
            elif token in punctuation:
                """token is punctuation"""
                text_corrected.append(token)
            elif token.isnumeric():
                """token is numeric"""
                text_corrected.append(token)
            elif self.___is_in_morfeusz_dict(token):
                """token already exists in polish dictionary"""
                text_corrected.append(token)
            else:
                if self.__check_if_token_is_upper(token):
                    token_correct = self.__correction_engine(token.lower()).upper()
                elif self.__check_if_token_is_capitalize(token):
                    token_correct = self.__correction_engine(token.lower()).capitalize()
                else:
                    token_correct = self.__correction_engine(token.lower())
                text_corrected.append(token_correct)
        return "".join(text_corrected)
