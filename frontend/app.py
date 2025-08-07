import streamlit as st
import numpy as np

def main():
    st.title("An app with a graph")

    # Generate some random data
    data = np.random.randn(100, 2)
    # Create a scatter plot
    st.subheader("Random Scatter Plot")
    st.write("This is a simple scatter plot of random data points.")
    st.scatter_chart(data)

if __name__ == "__main__":
    main()
