from nlp.tokenizer import *

import unittest2 as unittest

class TestStringMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_punct1(self):
        sentence = "What is the story of Kohinoor (Koh-i-Noor) Diamond?"
        output = " ".join(t.text for t in SpacyTokens(sentence).remove(punct))
        expected = "what story kohinoor koh-i-noor diamond"
        self.assertEqual(expected, output)

    def test_punct2(self):
        sentence = "What does it feel like to have sex with a man if you are (or were) a straight man?"
        output = " ".join(t.text for t in SpacyTokens(sentence).remove(punct))
        expected = "what feel like sex man straight man"
        self.assertEqual(expected, output)

    def test_punct3(self):
        sentence = "I'm a boy. I masturbate. I'm 13. Is it bad to masturbate?"
        output = " ".join(t.text for t in SpacyTokens(sentence).remove(punct))
        expected = "i be boy i masturbate i be is bad masturbate"
        self.assertEqual(expected, output)

    def test_punct4(self):
        sentence = "What is the difference between urban and rural living in the U.S.?"
        output = " ".join(t.text for t in SpacyTokens(sentence).remove(punct))
        expected = "what difference urban rural live u s"
        self.assertEqual(expected, output)

    def test_punct5(self):
        sentence = "How do I build and host an e-commerce website?"
        output = " ".join(t.text for t in SpacyTokens(sentence).remove(punct))
        expected = "how i build host e commerce website"
        self.assertEqual(expected, output)

    def test_stop1(self):
        sentence = "Is black or white the absence of color?"
        output = " ".join(t.text for t in SpacyTokens(sentence).remove(punct))
        expected = "black white absence color"
        self.assertEqual(expected, output)

    def test_numbers1(self):
        sentence = "Who will win the 2016 presidential elections?"
        output = " ".join(t.text for t in SpacyTokens(sentence).remove(punct))
        expected = "who win 2016 presidential election"
        self.assertEqual(expected, output)

    def test_question1(self):
        sentence = "Who will win the 2016 presidential elections?"
        output = " ".join(t.text for t in SpacyTokens(sentence).remove(punct))
        expected = "who win 2016 presidential election"
        self.assertEqual(expected, output)

    def test_question2(self):
        sentence = "Which phone should I buy under 15k?"
        output = " ".join(t.text for t in SpacyTokens(sentence).remove(punct))
        expected = "which phone I buy under 15k"
        self.assertEqual(expected, output)

    def test_question3(self):
        sentence = "What are the Best B2B website?"
        output = " ".join(t.text for t in SpacyTokens(sentence).remove(punct))
        expected = "what best b2b website"
        self.assertEqual(expected, output)

    def test_question4(self):
        sentence = "I really like this guy. He has a girlfriend, but we talk all the time and it seems like he flirts with me. How do I get him to like me?"
        output = " ".join(t.text for t in SpacyTokens(sentence).remove(punct))
        expected = "i really like guy he girlfriend talk time seem like flirt how i get like"
        self.assertEqual(expected, output)

    def test_question5(self):
        sentence = "Which hindi songs can be played using A D C Em chords?"
        output = " ".join(t.text for t in SpacyTokens(sentence).remove(punct))
        expected = "which hindi song play use a d c em chord"
        self.assertEqual(expected, output)

    def test_question6(self):
        sentence = "What is the science behind this? Why does the coaster not fall down?"
        output = " ".join(t.text for t in SpacyTokens(sentence).remove(punct))
        expected = "what science behind why coaster fall"
        self.assertEqual(expected, output)

    def test_question7(self):
        sentence = "Do employees at Lowe's have a good work-life balance? Does this differ across positions and departments?"
        output = " ".join(t.text for t in SpacyTokens(sentence).remove(punct))
        expected = "do employee lowe s good work life balance does differ across position department"
        self.assertEqual(expected, output)



if __name__ == '__main__':
    unittest.main()