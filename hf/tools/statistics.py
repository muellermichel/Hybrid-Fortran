from tools.commons import enum
import logging

Counters = enum(
    "DELETED_FOR_HF",
    "DELETED_WITHOUT_HF",
    "CHANGE_FOR_ORDERING",
    "ADDED_FOR_HF",
    "ADDED_WITHOUT_HF",
    "UNCHANGED",
    "ALL"
)

class Statistics(object):
    def __init__(self):
        self.reset()

    def __repr__(self):
        name_by_index = {}
        for name, number in Counters.__dict__.items():
            if not isinstance(number, int):
                continue
            name_by_index[number] = name.lower()

        return "Statistics(\n  Overall:\n%s\n  By Module:\n%s\n)" %(
            ",\n".join([
                "    %s: %i" %(name_by_index[c], self.counters[c])
                for c in self.counters
            ]),
            ",\n".join([
                "    %s:\n%s" %(
                    m,
                    ",\n".join([
                        "      %s: %i" %(name_by_index[c], self.counters_by_module_name[m][c])
                        for c in self.counters_by_module_name[m]
                    ])
                )
                for m in self.counters_by_module_name
            ])
        )

    def reset(self):
        self.counters = {}
        self.counters_by_module_name = {}
        logging.info("statistics reset")

    def addToCounter(self, counter, moduleName, loc=1):
        assert isinstance(counter, int)
        assert isinstance(moduleName, (str, unicode))
        assert len(moduleName) > 0
        assert isinstance(loc, int)
        if moduleName in ["helper_functions", "helper_functions_cuda_fortran", "helper_functions_gpu"]:
            return

        if not counter in self.counters:
            self.counters[counter] = 0
        if not moduleName in self.counters_by_module_name:
            self.counters_by_module_name[moduleName] = {}
        if not counter in self.counters_by_module_name[moduleName]:
            self.counters_by_module_name[moduleName][counter] = 0

        self.counters[counter] += loc
        self.counters_by_module_name[moduleName][counter] += loc

statistics = Statistics()