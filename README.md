# word2vec_test_language
A simple toy language to explore semantic vector training

main.py contains the example code to create and simulate a toy language with prespecified "semantic" relationships. The idea is explained here: https://www.tegladwin.com/files/howto/word2vec_toy_language.php.

During simulations, the language generator pseudorandomly pick a word-pair (the "throw" of dice variable), and one the pair is selected to be used in a sentence. The surrounding words are determined by the word-pair selected (depending on the noise parameter, this can be partly random). A set of similarities between selected pairs of words are calculated for each run of the simulation, and for each of a range of training times. The basic test is whether words within the same pair are more similar then words taken from different pairs. The pairs {"0", "1"} and {"2", "3"} are used for this.

There is also an overlapping pair of pairs, {"4", "5"} and {"4", "7"}. This is used to test a biasing effect, created by adding echoes of the {"4", "5"} pair but not of the {"4", "7"} pair.

(Not made to be in any way efficient - just as simple and transparent as possible as a basis to try stuff out and see what's going on!)


[![DOI](https://zenodo.org/badge/583027151.svg)](https://zenodo.org/badge/latestdoi/583027151)

