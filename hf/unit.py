import unittest

def tupleFromMatch(match):
	if not match:
		return ()
	return match.groups()

class Test_Decorators(unittest.TestCase):
	def test_decorator_lru_cache(self):
		import time
		import random
		from tools.commons import lru_cache

		class LRU_Test(object):
			"""class"""
			def __init__(self):
				self.num = 0

			@lru_cache(maxsize = 10, timeout = 3)
			def test_method(self, num):
				"""test_method_doc"""
				self.num += num
				return self.num

		@lru_cache(maxsize = 10, timeout = 3)
		def test_func(num):
			"""test_func_doc"""
			return num

		@lru_cache(maxsize = 10, timeout = 3)
		def test_func_time(num):
			"""test_func_time_doc"""
			return time.time()

		@lru_cache(maxsize = 10, timeout = None)
		def test_func_args(*args, **kwargs):
			return random.randint(1,10000000)

		# Init vars:
		c1 = LRU_Test()
		c2 = LRU_Test()
		m1 = c1.test_method
		m2 = c2.test_method
		f1 = test_func

		# Test basic caching functionality:
		self.assertEqual(m1(1), m1(1)) 
		self.assertEqual(c1.num, 1)     # c1.num now equals 1 - once cached, once real
		self.assertEqual(f1(1), f1(1))

		# Test caching is different between instances - once cached, once not cached:
		self.assertNotEqual(m1(2), m2(2))
		self.assertNotEqual(m1(2), m2(2))

		# Validate the cache_clear funcionality only on one instance:
		prev1 = m1(1)
		prev2 = m2(1)
		prev3 = f1(1)
		m1.cache_clear()
		self.assertNotEqual(m1(1), prev1)
		self.assertEqual(m2(1), prev2)
		self.assertEqual(f1(1), prev3)

		# Validate the docstring and the name are set correctly:
		self.assertEqual(m1.__doc__, "test_method_doc")
		self.assertEqual(f1.__doc__, "test_func_doc")
		self.assertEqual(m1.__name__, "test_method")
		self.assertEqual(f1.__name__, "test_func")

		# Test the limit of the cache, cache size is 10, fill 15 vars, the first 5 will be overwritten for each and the other 5 are untouched. Test that:
		c1.num = 0
		c2.num = 10
		m1.cache_clear()
		m2.cache_clear()
		f1.cache_clear()
		temp_list = map(lambda i : (test_func_time(i), m1(i), m2(i)), range(15))

		for i in range(5, 10):
			self.assertEqual(temp_list[i], (test_func_time(i), m1(i), m2(i)))
		for i in range(0, 5):
			self.assertNotEqual(temp_list[i], (test_func_time(i), m1(i), m2(i)))
		# With the last run the next 5 vars were overwritten, now it should have only 0..4 and 10..14:
		for i in range(5, 10):
			self.assertNotEqual(temp_list[i], (test_func_time(i), m1(i), m2(i)))

		# Test different vars don't collide:
		self.assertNotEqual(test_func_args(1), test_func_args('1'))
		self.assertNotEqual(test_func_args(1.0), test_func_args('1.0'))
		self.assertNotEqual(test_func_args(1.0), test_func_args(1))
		self.assertNotEqual(test_func_args(None), test_func_args('None'))
		self.assertEqual(test_func_args(test_func), test_func_args(test_func))
		self.assertEqual(test_func_args(LRU_Test), test_func_args(LRU_Test))
		self.assertEqual(test_func_args(object), test_func_args(object))
		self.assertNotEqual(test_func_args(1, num = 1), test_func_args(1, num = '1'))
		# Test the sorting of kwargs:
		self.assertEqual(test_func_args(1, aaa = 1, bbb = 2), test_func_args(1, bbb = 2, aaa = 1))
		self.assertNotEqual(test_func_args(1, aaa = '1', bbb = 2), test_func_args(1, bbb = 2, aaa = 1))


		# Sanity validation of values
		c1.num = 0
		c2.num = 10
		m1.cache_clear()
		m2.cache_clear()
		f1.cache_clear()
		self.assertEqual((f1(0), m1(0), m2(0)), (0, 0, 10))
		self.assertEqual((f1(0), m1(0), m2(0)), (0, 0, 10))
		self.assertEqual((f1(1), m1(1), m2(1)), (1, 1, 11))
		self.assertEqual((f1(2), m1(2), m2(2)), (2, 3, 13))
		self.assertEqual((f1(2), m1(2), m2(2)), (2, 3, 13))
		self.assertEqual((f1(3), m1(3), m2(3)), (3, 6, 16))
		self.assertEqual((f1(3), m1(3), m2(3)), (3, 6, 16))
		self.assertEqual((f1(4), m1(4), m2(4)), (4, 10, 20))
		self.assertEqual((f1(4), m1(4), m2(4)), (4, 10, 20))

		# Test timeout - sleep, it should refresh cache, and then check it was cleared:
		prev_time = test_func_time(0)
		self.assertEqual(test_func_time(0), prev_time)
		self.assertEqual(m1(4), 10)
		self.assertEqual(m2(4), 20)
		time.sleep(3.5)
		self.assertNotEqual(test_func_time(0), prev_time)
		self.assertNotEqual(m1(4), 10)
		self.assertNotEqual(m2(4), 20)

class TestPatterns(unittest.TestCase):
	def testImportPatterns(self):
		from tools.patterns import regexPatterns
		patterns = regexPatterns
		self.assertEqual(
			tupleFromMatch(patterns.importPattern.match("useme, only: some, distraction")),
			()
		)
		self.assertEqual(
			tupleFromMatch(patterns.importPattern.match("use my_module")),
			()
		)
		self.assertEqual(
			tupleFromMatch(patterns.importPattern.match("use my_module, only: a, ab, ba")),
			("my_module", "a, ab, ba")
		)
		self.assertEqual(
			tupleFromMatch(patterns.importPattern.match("use my_module, only: a, ab=>my_ba, ba")),
			("my_module", "a, ab=>my_ba, ba")
		)
		self.assertEqual(
			tupleFromMatch(patterns.importPattern.match("use my_module, only: a, ab => my_ba, ba")),
			("my_module", "a, ab => my_ba, ba")
		)

class TestCommonTools(unittest.TestCase):
	def testTextSplittingBasic(self):
		from tools.commons import splitTextAtLeftMostOccurrence
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
		self.assertEqual(
			splitTextAtLeftMostOccurrence("l", "cll(:,:) = cll_hfdev(:,:)"),
			("cll(:,:) = cll_hfdev(:,:)", "", "")
		)

	def testTextSplittingQuoted(self):
		from tools.commons import splitTextAtLeftMostOccurrence
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

	def testTextSplittingIntoComponents(self):
		from tools.commons import splitIntoComponentsAndRemainder
		components, remainder = splitIntoComponentsAndRemainder("")
		self.assertTrue(len(components) == 0)
		self.assertTrue(remainder == "")

		components, remainder = splitIntoComponentsAndRemainder("  ")
		self.assertTrue(len(components) == 0)
		self.assertEqual(remainder, "")

		components, remainder = splitIntoComponentsAndRemainder("a")
		self.assertEqual(
			tuple(components),
			("a",)
		)
		self.assertEqual(remainder, "")

		components, remainder = splitIntoComponentsAndRemainder("a b")
		self.assertEqual(
			tuple(components),
			("a",)
		)
		self.assertEqual(remainder, "b")

		components, remainder = splitIntoComponentsAndRemainder("a  b")
		self.assertEqual(
			tuple(components),
			("a",)
		)
		self.assertEqual(remainder, "b")

		components, remainder = splitIntoComponentsAndRemainder("a, b b")
		self.assertEqual(
			tuple(components),
			("a", "b")
		)
		self.assertEqual(remainder, "b")

		components, remainder = splitIntoComponentsAndRemainder("a , b b")
		self.assertEqual(
			tuple(components),
			("a", "b")
		)
		self.assertEqual(remainder, "b")

		components, remainder = splitIntoComponentsAndRemainder("a, b(n, m) b")
		self.assertEqual(
			tuple(components),
			("a", "b(n, m)")
		)
		self.assertEqual(remainder, "b")

		components, remainder = splitIntoComponentsAndRemainder("a, b(n * (m + 1)) b")
		self.assertEqual(
			tuple(components),
			("a", "b(n * (m + 1))")
		)
		self.assertEqual(remainder, "b")

		components, remainder = splitIntoComponentsAndRemainder("a, b, b(n * (m + 1))")
		self.assertEqual(
			tuple(components),
			("a", "b", "b(n * (m + 1))")
		)
		self.assertEqual(remainder, "")

		components, remainder = splitIntoComponentsAndRemainder("a, b(n * (m + 1)), b")
		self.assertEqual(
			tuple(components),
			("a", "b(n * (m + 1))", "b")
		)
		self.assertEqual(remainder, "")

		components, remainder = splitIntoComponentsAndRemainder("a,b(n * (m + 1)),b")
		self.assertEqual(
			tuple(components),
			("a", "b(n * (m + 1))", "b")
		)
		self.assertEqual(remainder, "")

		components, remainder = splitIntoComponentsAndRemainder("a, b ::b")
		self.assertEqual(
			tuple(components),
			("a", "b")
		)
		self.assertEqual(remainder, "::b")

class TestMachineryAlgorithms(unittest.TestCase):
	def testSpecificationParsing(self):
		from machinery.commons import parseSpecification
		self.assertEqual(
			parseSpecification(""),
			(None, None, None)
		)
		self.assertEqual(
			parseSpecification("blub blib"),
			(None, None, None)
		)
		self.assertEqual(
			parseSpecification("real a"),
			("real", (("a", None),), "")
		)
		self.assertEqual(
			parseSpecification("real, attribute a"),
			("real, attribute", (("a", None),), "")
		)
		self.assertEqual(
			parseSpecification("real, attribute a(m, n)"),
			("real, attribute", (("a", "m, n"),), "")
		)
		self.assertEqual(
			parseSpecification("real, attribute a(m * (n + 1))"),
			("real, attribute", (("a", "m * (n + 1)"),), "")
		)
		self.assertEqual(
			parseSpecification("real, attribute a (m * (n + 1))"),
			("real, attribute", (("a", "m * (n + 1)"),), "")
		)
		self.assertEqual(
			parseSpecification("real, attribute(m, n) a"),
			("real, attribute(m, n)", (("a", None),), "")
		)
		self.assertEqual(
			parseSpecification("real, attribute(m * (n + 1)) a"),
			("real, attribute(m * (n + 1))", (("a", None),), "")
		)
		self.assertEqual(
			parseSpecification("real, attribute (m * (n + 1)) a"),
			("real, attribute (m * (n + 1))", (("a", None),), "")
		)
		self.assertEqual(
			parseSpecification("real, attribute :: a, b"),
			("real, attribute", (("a", None), ("b", None)), "")
		)
		self.assertEqual(
			parseSpecification("real, attribute :: a(m, n), b"),
			("real, attribute", (("a", "m, n"), ("b", None)), "")
		)
		self.assertEqual(
			parseSpecification("real, attribute a = 1.0d0"),
			("real, attribute", (("a", None),), "= 1.0d0")
		)
		self.assertEqual(
			parseSpecification("real, attribute a= 1.0d0"),
			("real, attribute", (("a", None),), "= 1.0d0")
		)
		self.assertEqual(
			parseSpecification("real, attribute :: a= 1.0d0"),
			("real, attribute", (("a", None),), "= 1.0d0")
		)
		self.assertEqual(
			parseSpecification("real, attribute::a= 1.0d0"),
			("real, attribute", (("a", None),), "= 1.0d0")
		)
		self.assertEqual(
			parseSpecification("double precision a"),
			("double precision", (("a", None),), "")
		)
		self.assertEqual(
			parseSpecification("double precision :: a"),
			("double precision", (("a", None),), "")
		)
		self.assertEqual(
			parseSpecification("double precision, attribute :: a = 1.0d0"),
			("double precision, attribute", (("a", None),), "= 1.0d0")
		)
		self.assertEqual(
			parseSpecification("double precision, attribute a = 1.0d0"),
			("double precision, attribute", (("a", None),), "= 1.0d0")
		)

class TestSymbolAlgorithms(unittest.TestCase):
	def testSymbolNamesFromDeclaration(self):
		def symbolNamesFromDeclaration(declaration):
			from models.symbol import symbolNamesFromSpecificationTuple
			from machinery.commons import parseSpecification
			return symbolNamesFromSpecificationTuple(
				parseSpecification(declaration)
			)
		self.assertEqual(
			symbolNamesFromDeclaration("real :: a, b, c"),
			("a", "b", "c")
		)
		self.assertEqual(
			symbolNamesFromDeclaration("real :: a(n), b, c"),
			("a", "b", "c")
		)
		self.assertEqual(
			symbolNamesFromDeclaration("integer :: n, a(n), b, c"),
			("n", "a", "b", "c")
		)
		self.assertEqual(
			symbolNamesFromDeclaration("integer :: a(n, m), b, c"),
			("a", "b", "c")
		)
		self.assertEqual(
			symbolNamesFromDeclaration("integer :: a(n * (m + k)), b, c"),
			("a", "b", "c")
		)

	def testDimensionString(self):
		def dimensionStringFromDeclaration(symbolName, declaration):
			from models.symbol import dimensionStringFromSpecification
			from machinery.commons import parseSpecification
			return dimensionStringFromSpecification(
				symbolName,
				parseSpecification(declaration)
			)
		self.assertEqual(
			dimensionStringFromDeclaration("a", "real :: a, b, c"),
			None
		)
		self.assertEqual(
			dimensionStringFromDeclaration("a", "real :: a, b(n, m), c"),
			None
		)
		self.assertEqual(
			dimensionStringFromDeclaration("b", "real :: a, b(n, m), c"),
			"n, m"
		)
		self.assertEqual(
			dimensionStringFromDeclaration("a", "real, dimension(n, m) :: a, b, c"),
			"n, m"
		)
		self.assertEqual(
			dimensionStringFromDeclaration("b", "real, dimension(n, m) :: a, b, c"),
			"n, m"
		)
		self.assertEqual(
			dimensionStringFromDeclaration("b", "real, dimension(n * (m + k)) :: a, b, c"),
			"n * (m + k)"
		)
		self.assertEqual(
			dimensionStringFromDeclaration("a", "real :: a(n, m), b, c"),
			"n, m"
		)
		self.assertEqual(
			dimensionStringFromDeclaration("a", "real :: a(n * (m + k)), b, c"),
			"n * (m + k)"
		)
		self.assertEqual(
			dimensionStringFromDeclaration("a", "real a(n, m)"),
			"n, m"
		)
		self.assertEqual(
			dimensionStringFromDeclaration("a", "real a(n * (m + k))"),
			"n * (m + k)"
		)
		self.assertEqual( #testing whether a symbol being a prefix of another is handled correclty
			dimensionStringFromDeclaration("a", "real :: ab, a(n, m)"),
			"n, m"
		)
		self.assertEqual( #testing whether a symbol being a suffix of another is handled correclty
			dimensionStringFromDeclaration("a", "real :: ba, a(n, m)"),
			"n, m"
		)
		self.assertEqual(
			dimensionStringFromDeclaration("a", "real :: ab,a(n, m)"),
			"n, m"
		)
		self.assertEqual(
			dimensionStringFromDeclaration("a", "real :: ba,a(n, m)"),
			"n, m"
		)

	def testsplitAndPurgeSpecification(self):
		from models.symbol import splitAndPurgeSpecification
		self.assertEqual(
			splitAndPurgeSpecification(
				"real, intent(in) :: a, b",
				purgeList=['intent']
			),
			("real", "real, intent(in)", "a, b")
		)
		self.assertEqual(
			splitAndPurgeSpecification(
				"real, intent(in) a",
				purgeList=['intent']
			),
			("real", "real, intent(in)", "a")
		)
		self.assertEqual(
			splitAndPurgeSpecification(
				"real(8), intent(in) :: a, b",
				purgeList=['intent']
			),
			("real(8)", "real(8), intent(in)", "a, b")
		)
		self.assertEqual(
			splitAndPurgeSpecification(
				"real(8), intent(in) a",
				purgeList=['intent']
			),
			("real(8)", "real(8), intent(in)", "a")
		)
		self.assertEqual(
			splitAndPurgeSpecification(
				"real(8), dimension(n * (m + 1)) :: a, b",
				purgeList=['intent', 'dimension']
			),
			("real(8)", "real(8), dimension(n * (m + 1))", "a, b")
		)
		self.assertEqual(
			splitAndPurgeSpecification(
				"real(8), dimension(n * (m + 1)) a",
				purgeList=['intent', 'dimension']
			),
			("real(8)", "real(8), dimension(n * (m + 1))", "a")
		)
		self.assertEqual(
			splitAndPurgeSpecification(
				"real, parameter :: my_parameter = 1.0d0",
				purgeList=[]
			),
			("real, parameter", "real, parameter", "my_parameter = 1.0d0")
		)

class TestImplementationAlgorithms(unittest.TestCase):
	def testRoutineNameSynthesis(self):
		from implementations.commons import synthesizedKernelName
		from implementations.commons import synthesizedHostRoutineName
		from implementations.commons import synthesizedDeviceRoutineName
		self.assertEqual(
			synthesizedKernelName("a", 1),
			"hfk1_a"
		)
		self.assertEqual(
			synthesizedKernelName("_a", 1),
			"hfk1__a"
		)
		self.assertEqual(
			synthesizedKernelName("hfk1_a", 1),
			"hfk1_a"
		)
		self.assertEqual(
			synthesizedKernelName("hfd_a", 1),
			"hfk1_a"
		)
		self.assertEqual(
			synthesizedHostRoutineName("_a"),
			"_a"
		)
		self.assertEqual(
			synthesizedHostRoutineName("hfd_a"),
			"a"
		)
		self.assertEqual(
			synthesizedHostRoutineName("hfk1_a"),
			"a"
		)
		self.assertEqual(
			synthesizedDeviceRoutineName("_a"),
			"hfd__a"
		)
		self.assertEqual(
			synthesizedDeviceRoutineName("hfd_a"),
			"hfd_a"
		)
		self.assertEqual(
			synthesizedDeviceRoutineName("hfk1_a"),
			"hfd_a"
		)

class TestPickling(unittest.TestCase):
	def makeDummyXMLString(self):
		import re

		dummyCallGraphXML = re.sub(r"[\n\t]*", "", """
			<callGraph>
				<routines>
					<routine module="foo" name="bar" source="foo">
						<domainDependants>
							<templateRelation id="testTemplateID">
								<entry>testSymbol</entry>
							</templateRelation>
						</domainDependants>
					</routine>
				</routines>
				<modules>
					<module name="foo"/>
				</modules>
				<domainDependantTemplates>
					<domainDependantTemplate id="testTemplateID">
						<attribute>
							<entry>present</entry>
							<entry>autoDom</entry>
						</attribute>
						<domName>
							<entry>x</entry>
							<entry>y</entry>
							<entry>k</entry>
						</domName>
						<domSize>
							<entry>nx</entry>
							<entry>ny</entry>
							<entry>nz</entry>
						</domSize>
					</domainDependantTemplate>
				</domainDependantTemplates>
			</callGraph>
		""")
		return dummyCallGraphXML

	def makeDummyCallGraphDocument(self):
		from tools.metadata import parseString
		
		return parseString(self.makeDummyXMLString(), immutable=False)

	def makeDummyModule(self, cgDoc):
		from models.module import Module

		moduleNode = cgDoc.firstChild.childNodes[1].firstChild
		self.assertEqual(moduleNode.tagName, "module")
		return Module(
			"foo",
			moduleNode
		)

	def makeDummyRoutine(self, cgDoc, module):
		from implementations.fortran import FortranImplementation

		routineNode = cgDoc.firstChild.firstChild.firstChild
		self.assertEqual(routineNode.tagName, "routine")
		return module.createRoutine(
			"bar",
			routineNode,
			{},
			FortranImplementation()
		)

	def makeDummySymbol(self, cgDoc, module, routine):
		from models.symbol import Symbol
		from tools.metadata import getDomainDependantTemplatesAndEntries

		routineNode = routine.node
		self.assertEqual(routineNode.tagName, "routine")

		templatesAndEntries = getDomainDependantTemplatesAndEntries(cgDoc, routineNode)
		self.assertEqual(len(templatesAndEntries), 1)

		template, entry = templatesAndEntries[0]
		symbol = Symbol(
			"testSymbol",
			template,
			symbolEntry=entry,
			scopeNode=routineNode,
			analysis=None,
			parallelRegionTemplates=[],
			globalParallelDomainNames={}
		)
		routine.loadSymbolsByName({"testSymbol":symbol})
		return symbol

	def testCycleFreeDOMNode(self):
		from tools.metadata import CycleFreeDOMNode
		cycleFreeDoc = CycleFreeDOMNode(self.makeDummyCallGraphDocument())
		self.assertEqual(cycleFreeDoc.firstChild.tagName, "callGraph")
		self.assertEqual(cycleFreeDoc.firstChild.firstChild.tagName, "routines")
		self.assertEqual(cycleFreeDoc.firstChild.firstChild.firstChild.tagName, "routine")
		self.assertEqual(cycleFreeDoc.firstChild.firstChild.firstChild.getAttribute("module"), "foo")
		self.assertEqual(cycleFreeDoc.firstChild.firstChild.firstChild.getAttribute("name"), "bar")

		domainDependantTemplateNodes = cycleFreeDoc.getElementsByTagName("domainDependantTemplate")
		self.assertEqual(len(domainDependantTemplateNodes), 1)
		attributeNodes = domainDependantTemplateNodes[0].getElementsByTagName("attribute")
		self.assertEqual(len(attributeNodes), 1)
		self.assertEqual(len(attributeNodes[0].childNodes), 2)
		self.assertEqual(attributeNodes[0].childNodes[1].firstChild.nodeValue, "autoDom")

	def testCallGraphPickling(self):
		from tools.metadata import CycleFreeDOMNode
		import pickle

		#just test whether pickling doesn't throw an error here
		_ = pickle.loads(pickle.dumps(self.makeDummyCallGraphDocument()))
		_ = pickle.loads(pickle.dumps(CycleFreeDOMNode(self.makeDummyCallGraphDocument())))

	def testModulePickling(self):
		import pickle

		cgDoc = self.makeDummyCallGraphDocument()
		module = self.makeDummyModule(cgDoc)

		# in order to implement concurrency using multiprocessing,
		# we want the following to work:
		_ = pickle.loads(pickle.dumps(module))

	def testRoutinePickling(self):
		import pickle

		cgDoc = self.makeDummyCallGraphDocument()
		module = self.makeDummyModule(cgDoc)
		routine = self.makeDummyRoutine(cgDoc, module)

		# in order to implement concurrency using multiprocessing,
		# we want the following to work:
		_ = pickle.loads(pickle.dumps(routine))

	def testSymbolPickling(self):
		import pickle

		cgDoc = self.makeDummyCallGraphDocument()
		module = self.makeDummyModule(cgDoc)
		routine = self.makeDummyRoutine(cgDoc, module)
		symbol = self.makeDummySymbol(cgDoc, module, routine)
		self.assertEqual(symbol.name, "testSymbol")
		self.assertEqual(len(symbol.domains), 3)

		# in order to implement concurrency using multiprocessing,
		# we want the following to work:
		_ = pickle.loads(pickle.dumps(routine))

		symbolAfterPickling = pickle.loads(pickle.dumps(symbol))
		self.assertEqual(symbolAfterPickling.name, "testSymbol")
		self.assertEqual(len(symbolAfterPickling.domains), 3)

if __name__ == '__main__':
	unittest.main()