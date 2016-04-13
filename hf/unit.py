import unittest
from tools.commons import splitTextAtLeftMostOccurrence

class TestCommonTools(unittest.TestCase):

	def testTextSplittingBasic(self):
		self.assertEqual(
			splitTextAtLeftMostOccurrence("", ""),
			("", "", "")
		)
		self.assertEqual(
			splitTextAtLeftMostOccurrence("abcd", ""),
			("", "", "")
		)
		self.assertEqual(
			splitTextAtLeftMostOccurrence("", "abcd"),
			("abcd", "", "")
		)
		self.assertEqual(
			splitTextAtLeftMostOccurrence("a", "abcd"),
			("abcd", "", "")
		)
		self.assertEqual(
			splitTextAtLeftMostOccurrence("a", "a bcd"),
			("", "a", " bcd")
		)
		self.assertEqual(
			splitTextAtLeftMostOccurrence("a", "a bcda"),
			("", "a", " bcda")
		)
		self.assertEqual(
			splitTextAtLeftMostOccurrence("a", " bcda"),
			(" bcda", "", "")
		)
		self.assertEqual(
			splitTextAtLeftMostOccurrence("a", " bcd a"),
			(" bcd ", "a", "")
		)
		self.assertEqual(
			splitTextAtLeftMostOccurrence("a", "bcd a123"),
			("bcd a123", "", "")
		)
		self.assertEqual(
			splitTextAtLeftMostOccurrence("a", "bcd a 123"),
			("bcd ", "a", " 123")
		)

	def testTextSplittingQuoted(self):
		self.assertEqual(
			splitTextAtLeftMostOccurrence("a", "bcd \"a\" 123"),
			("bcd \"a\" 123", "", "")
		)
		self.assertEqual(
			splitTextAtLeftMostOccurrence("a", "bcd \'a\' 123"),
			("bcd \'a\' 123", "", "")
		)
		self.assertEqual(
			splitTextAtLeftMostOccurrence("a", "bcd \'a\'a 123"),
			("bcd \'a\'", "a", " 123")
		)
		self.assertEqual(
			splitTextAtLeftMostOccurrence("a", "\'a\'a 123"),
			("\'a\'", "a", " 123")
		)
		self.assertEqual(
			splitTextAtLeftMostOccurrence("a", "bcd a\'a\'"),
			("bcd ", "a", "\'a\'")
		)
		#$$$ quotes within quotes analysis not supported for now
		# self.assertEqual(
		# 	splitTextAtLeftMostOccurrence("a", "bcd \"a\'a\'\"a"),
		# 	("bcd \"a\'a\'\"", "a", "")
		# )
		# self.assertEqual(
		# 	splitTextAtLeftMostOccurrence("a", "bcd \"a\'a\'\"a 123"),
		# 	("bcd \"a\'a\'\"", "a", " 123")
		# )
		# self.assertEqual(
		# 	splitTextAtLeftMostOccurrence("a", "bcd \"a\'a\'\" \'blub a\'a 123"),
		# 	("bcd \"a\'a\'\" \'blub a\'", "a", " 123")
		# )

if __name__ == '__main__':
	unittest.main()