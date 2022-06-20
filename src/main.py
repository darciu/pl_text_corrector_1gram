# -*- coding: utf-8 -*-
import re
import pickle
import gzip
from string import punctuation
import unicodedata
import itertools
from typing import List

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
            "ę": ["w", "s", "d", "r"],
            "r": ["e", "d", "f", "t"],
            "t": ["r", "f", "g", "y"],
            "y": ["t", "g", "h", "u"],
            "u": ["y", "h", "j", "i"],
            "i": ["u", "j", "k", "o"],
            "o": ["i", "k", "l", "p"],
            "ó": ["i", "k", "l", "p"],
            "p": ["o", "l"],
            "a": ["q", "w", "s", "z"],
            "ą": ["q", "w", "s", "z"],
            "s": ["a", "w", "e", "d", "x", "z"],
            "ś": ["a", "w", "e", "d", "x", "z"],
            "d": ["s", "e", "r", "f", "c", "x"],
            "f": ["d", "r", "t", "g", "v", "c"],
            "g": ["f", "t", "y", "h", "b", "v"],
            "h": ["g", "y", "u", "j", "n", "b"],
            "j": ["h", "u", "i", "k", "m", "n"],
            "k": ["j", "i", "o", "l", "m"],
            "l": ["k", "o", "p"],
            "ł": ["k", "o", "p"],
            "z": ["a", "s", "x"],
            "ż": ["a", "s", "x"],
            "ź": ["a", "s", "x"],
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

    def __remove_xd(self, token: str) -> str:
        if token[-2:] == 'xd':
            return token[:-2]
        return token

    def __reduce_repeated_letters(self, token: str) -> str:
        return re.sub(r'([a-z])\1+', r'\1', token)

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
        """gather all diacritic token variations and return list with no duplicates"""

        # # remove diacritics
        def strip_accents(s):
            return ''.join(c for c in unicodedata.normalize('NFD', s)
                            if unicodedata.category(c) != 'Mn')

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
            'd': 't',
            't': 'd',
            'w': 'f',
            'f': 'w',
            'z': 's',
            's': 'z'
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

    def __phonetic_combinations(self, token: str) -> list:
        """gather all phonetic token variations and return list with no duplicates"""

        all_combinations = set([token])
        for index in range(len(token)):
            all_combinations.update(
                self.__replace_with_phonetic(all_combinations, index)
            )
        return list(all_combinations)
        
    def __all_elements_combinations(self, lst: list) -> List[List[tuple]]:
        """retrun all combinations of elements for given list"""
        return [list(itertools.combinations(lst, i)) for i in range(1,len(lst)+1)]

    def __spelling_combinations(self, token: str,tup: tuple) -> list:
        """return combinations of given token according to provided spelling correction"""
        len_1 = len(tup[0])
        len_2 = len(tup[1])
        if len_1 > len_2:
            token = token.replace(tup[0],tup[0] + ' '*(len_2-len_1))
            tup = (tup[0] + ' '*(len_2-len_1),tup[1])
            len_1 = len(tup[0])
            len_2 = len(tup[1])
        token_combinations = list()
        for comb in self.__all_elements_combinations([m.start() for m in re.finditer(tup[0], token)]):
            for perm in comb:
                token_copy = token
                for pos in perm:
                    token_copy = token_copy[:pos] + token_copy[pos:pos+len_1].replace(tup[0],tup[1]+' '*(len_1-len_2)) + token_copy[pos+len_1:]
                token_combinations.append(token_copy.replace(' ',''))
        return token_combinations
    
    def __common_spelling_errors(self, token: str) -> List[str]:
        """return set of various word combination for most common spelling mistakes"""
        output_words = set()
        output_words.update([token])
        if token[-2:] == 'jo':
            output_words.update([token[:-2]+'ją'])
        elif token[-1] == 'i':
            output_words.update([token + 'i'])

        for p in [
                ("en","ę"),
                ("ua","ła"),
                ("ue","łe"),
                ("uy","ły"),
                ("om","ą"),
                ("ą","om"),
                ("on","ą"),
                ("rz","sz"),
                ("sz","rz"),
                ("rz","ż"),
                ("ż","rz"),
                ("ch","h"),
                ("h","ch"),
                ("u","ó"),
                ("ó","u"),
                ("cz","trz"),
                ("dz", "c"),
        ]:
            temp_set = set()
            for word in output_words:
                temp_set.update(self.__spelling_combinations(word,p))
            output_words = output_words.union(temp_set)
        return output_words

    def __remove_one_letter_from_token(self, token: str) -> list:
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

    def __qwerty_keyboard_typos(self, token: str) -> list:
        """
        return list of all token variations but every letter is replaced with one of qwerty
        keyboard typos (look at __qwerty_typos dictionary)
        """

        candidates = list()
        for index in range(len(token)):
            for typo in self.__qwerty_typos.get(token[index],[]):
                candidates.append(token[:index] + typo + token[index + 1 :])
        return candidates

    def __reduce_qwerty_keyboard_typo(self, token: str) -> list:
        """
        return list of all token variations but every possible qwerty keyboard double typo is reduced to one letter
        (eg. worlkd gives world and workd)
        """
        candidates = list()
        for index in range(len(token) - 1):
            if token[index] in self.__qwerty_typos.get(token[index + 1],[]):
                candidates.append(token[:index] + token[index] + token[index + 2 :])
                candidates.append(token[:index] + token[index + 1] + token[index + 2 :])
        return candidates

    def __additional_wildcard(self, token: str) -> list:
        """
        return list of all token variations but every two adjacent letters are filled with wildcard in between
        no wildcard at the end as it may indicate word's plural form
        """
        candidates = list()
        for index in range(len(token) - 1):
            candidates.append(token[:index] + "?" + token[index:])
        return candidates

    def __replace_letter_with_wildcard(self, token: str) -> list:
        """
        return list of all token variations but every letter is replaced with wildcard

        """
        candidates = list()
        for index in range(len(token)):
            candidates.append(token[:index] + "?" + token[index:])
        return candidates

    def __check_if_token_is_upper(self, token: str) -> bool:
        return token[0].isupper()

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

    def __construct_dataframe(self, candidates: list, weight: float) -> pd.DataFrame:
        """return pandas dataframe containing column with given candidates and weights"""
        df = pd.DataFrame(candidates)
        df.rename(columns={0:'token'},inplace=True)
        df["weight"] = weight
        return df

    def __wildcard_candidates_to_dataframe(
        self, df: pd.DataFrame, wildcard_candidates: list, weight: float
    ) -> pd.DataFrame:
        df = df[['token','weight','freq']]
        for candidate in wildcard_candidates:
            l = list(self.__words_trie.values(candidate, "?"))
            if len(l) > 0:
                for elem in l:

                    df.loc[len(df)] = [elem[1], weight, elem[0]]

        return df

    def __create_df_with_candidates(self, token: str) -> pd.DataFrame:
        diacritic_candidates = self.__diacritic_combinations(token)
        try: diacritic_candidates.remove(token)
        except: pass

        phonetic_candidates = self.__phonetic_combinations(token)
        try: phonetic_candidates.remove(token)
        except: pass

        spelling_errors_candidates = self.__common_spelling_errors(token)

        qwerty_typos_candidates = self.__qwerty_keyboard_typos(token)

        reduced_qwetry_typo_candidates = self.__reduce_qwerty_keyboard_typo(token)

        remove_one_letter_candidates = self.__remove_one_letter_from_token(token)

        swap_two_adjacent_letters_candidates = self.__swap_two_adjacent_letters(token)

        additional_wildcard_candidates = self.__additional_wildcard(token)

        replaced_letter_with_wildcard_candidates = self.__replace_letter_with_wildcard(
            token
        )

        df = (
            self.__construct_dataframe(diacritic_candidates, 0.5)
            .append(self.__construct_dataframe(phonetic_candidates, 1))
            .append(self.__construct_dataframe(spelling_errors_candidates, 1))
            .append(self.__construct_dataframe(qwerty_typos_candidates, 1))
            .append(self.__construct_dataframe(reduced_qwetry_typo_candidates, 1))
            .append(self.__construct_dataframe(remove_one_letter_candidates, 1))
            .append(
                self.__construct_dataframe(swap_two_adjacent_letters_candidates, 2.5)
            )
        )

        df["freq"] = df["token"].apply(self.__find_freq)

        df = self.__wildcard_candidates_to_dataframe(
            df, additional_wildcard_candidates, 2
        )
        df = self.__wildcard_candidates_to_dataframe(
            df, replaced_letter_with_wildcard_candidates, 6
        )

        df = df[df["freq"] > 0]

        return df

    def __correction_engine(self, token: str) -> str:
        """
        Main purpose of this part of the code is to collect as many resonable variations (candidates) for given token,
        find frequency of this variation in words_trie, calculate damerau-levenshtein distance from original token and variation
        and finally multiply everything including wages (there are different wages for different types of candicates).
        The top result with highest score will be returned as the most relevant token.
        """

        token = self.__remove_xd(token)

        #reduced_repeated_letters_candidate = [self.__reduce_repeated_letters(token),]

        df = self.__create_df_with_candidates(token)

        if len(df) > 0:
            df['token'] = df['token'].astype(str)
            df['token'] = df['token'].fillna('')
            df["calibrated_distance"] = (
            df["token"].apply(lambda x: damerau_levenshtein_distance(str(x), token)) + 3
            )  # this may be parameter
            df["result"] = df["freq"] / (df["weight"] + df["calibrated_distance"])
            df.sort_values(by="result", ascending=False, inplace=True, ignore_index=True)
            return df["token"].iloc[0]
        else:
            return token

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
                    token_correct = self.__correction_engine(token).capitalize()
                else:
                    token_correct = self.__correction_engine(token)
                text_corrected.append(token_correct)
        return "".join(text_corrected)
