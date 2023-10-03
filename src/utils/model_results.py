class ModelResults:
    def __init__(self, results):
        if not isinstance(results, dict):
            raise ValueError("TODO:", results)
        else:
            self.results = results

    def __str__(self):
        ret_str = ""
        for k, v in self.results.items():
            if ret_str:
                ret_str += " "

            ret_str += "{}: {:.4f}".format(k, v)

        return ret_str

    def __getitem__(self, key):
        return self.results[key]

    def __setitem__(self, key, value):
        self.results[key] = value
