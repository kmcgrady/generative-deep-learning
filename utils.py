import os
import re
import typing
import pandas as pd
import keras

if typing.TYPE_CHECKING:
    from keras import Model

# itch: Annoying way to get the model summary
def get_model_summary(model: "Model", debug=False):
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    summ_string = "\n".join(stringlist)
    if debug:
        print(summ_string) # entire summary in a variable

    table = stringlist[1:-4][1::2] # take every other element and remove appendix

    new_table = []
    for entry in table:
        entry = re.split(r'\s{2,}', entry)[:-1] # remove whitespace
        new_table.append(entry)

    return pd.DataFrame(new_table[1:], columns=new_table[0]).dropna(), stringlist[-4:-1]

# Write a decorator that takes in a name, looks inside a `.cache` folder
# if it exists, then load the keras model and return it. If it doesn't exist,
# then run the function and save the model to the `.cache` folder.
#
# The decorator should also take in a `debug` parameter that will print out
# the model summary if `True`.
# itch: This is needed to truly cache the model (and avoid retraining it every time)
def cache_model(name: str, *, debug: bool = False, cache_dir: str = ".cache"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            model = None
            try:
                model = keras.models.load_model(f"{cache_dir}/{name}.keras")
                if debug:
                    print("Loaded model from cache")
            except:
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                model = func(*args, **kwargs)
                model.save(os.path.join(cache_dir, f"{name}.keras"))
                if debug:
                    print("Saved model to cache")
            return model
        return wrapper
    
    return decorator

# itch: caching on the session state
def cache_session(name: str, *, debug: bool = False):
    import streamlit as st
    if "CACHE_SESSION" not in st.session_state:
        st.session_state.CACHE_SESSION = {}

    def decorator(func):
        def wrapper(*args, **kwargs):
            if name in st.session_state.CACHE_SESSION:
                if debug:
                    print("Loaded session from cache")
                return st.session_state.CACHE_SESSION[name]
            else:
                output = func(*args, **kwargs)
                st.session_state.CACHE_SESSION[name] = output
                if debug:
                    print("Saved session to cache")
                return output
        return wrapper
    
    return decorator

def clear_session_cache(name: str | None):
    import streamlit as st
    if name is None:
        st.session_state.CACHE_SESSION = {}
    else:
        del st.session_state.CACHE_SESSION[name]
