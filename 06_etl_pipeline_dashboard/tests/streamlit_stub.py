"""A minimal stand-in for the `streamlit` module so dashboard.py can be
executed top-to-bottom in tests (and in CI) without a browser. Every
widget returns its default; every output call records what it was given."""
import sys
import types

calls = []


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(module, name)


def _record(name):
    def fn(*args, **kwargs):
        calls.append((name, args, kwargs))
    return fn


def _default_widget(name):
    def fn(label, *args, **kwargs):
        calls.append((name, (label,), kwargs))
        if "default" in kwargs:
            return kwargs["default"]
        if "value" in kwargs:
            return kwargs["value"]
        return args[0] if args else None
    return fn


module = types.ModuleType("streamlit")
for name in ("set_page_config", "title", "caption", "header", "subheader", "metric", "line_chart",
             "bar_chart", "area_chart", "dataframe", "write", "error", "download_button"):
    setattr(module, name, _record(name))
module.multiselect = _default_widget("multiselect")
module.date_input = _default_widget("date_input")
module.checkbox = _default_widget("checkbox")
module.columns = lambda n: [_Ctx() for _ in range(n if isinstance(n, int) else len(n))]
module.tabs = lambda names: [_Ctx() for _ in names]
module.stop = lambda: (_ for _ in ()).throw(SystemExit("st.stop"))
module.cache_data = lambda *a, **k: (lambda f: f)
module.sidebar = _Ctx()
module.sidebar.header = _record("sidebar.header")


def install():
    calls.clear()
    sys.modules["streamlit"] = module
    return calls
