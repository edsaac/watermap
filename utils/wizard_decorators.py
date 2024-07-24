from typing import Callable, Optional
import streamlit as st


def next_step(func: Optional[Callable] = None):
    if func is None:
        st.session_state.wizard_step += 1
        return

    def wrapper():
        func()
        st.session_state.wizard_step += 1

    return wrapper


def previous_step(func: Optional[Callable] = None):
    if func is None:
        st.session_state.wizard_step -= 1
        return

    def wrapper():
        func()
        st.session_state.wizard_step -= 1

    return wrapper


def start_again(func: Optional[Callable] = None):
    if func is None:
        del st.session_state.wizard_step
        return

    def wrapper():
        func()
        del st.session_state.wizard_step

    return wrapper
