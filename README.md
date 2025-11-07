# MultiModalSentimentAnalysis

From scratch PyTorch implementation of a multimodal sentiment analysis deep learning model that predicts sentiment (positive, negative, neutral) from image + text pairs using the MVSA-multiple (Twitter) and TumEmo (Tumblr) datasets.

### Turing Setup Instructions

1. [Follow the Turing Getting Started guide](https://docs.turing.wpi.edu/getting-started/) for detailed setup instructions.
2. Once you have SSH access, clone this repository into your home directory on Turing:

    ```bash
    git clone https://github.com/BrentWeiffenbach/MultiModalSentimentAnalysis.git
    ```

3. Use the Turing OnDemand dashboard to upload the MVSA dataset. Place the dataset in the repository under the following path:

    ```
    /home/$USER/MultiModalSentimentAnalysis/MVSA
    ```

4. Open **Visual Studio Code**.
5. Install the **Remote - SSH** extension from the VS Code Extensions marketplace.
6. Click the green corner icon in the bottom-left of VS Code and select **"Remote-SSH: Connect to Host..."**.
7. Enter your SSH connection string (e.g., `username@turing.wpi.edu`) and connect.
8. To avoid entering your password each time, set up SSH key-based authentication:
    - Run `ssh-keygen` on your local machine and press Enter to accept defaults.
    - Copy your public key to the remote server with `ssh-copy-id username@turing.wpi.edu`.
    - Now, connecting via Remote Explorer in VS Code should not prompt for a password.
9. Use the **Remote Explorer** panel in VS Code and open MultiModalSentimentAnalysis.
10. In the remote VS Code session, install recommended extensions such as **Python** and **Pylance** from the Extensions marketplace for enhanced coding support.
11. Open a terminal in VS Code and run the following command to submit the job script, which will set up a virtual environment and run the dataset processing script:

    ```bash
    sbatch turing_test_script.sh
    ```

    This will execute `dataset_mvsa.py` and create a virtual environment (`mmsa_env`) for you.

12. Once the job is complete, select the newly created `mmsa_env` as your Python interpreter in VS Code. This will enable proper IntelliSense and code completion features while you work on the project.
13. Now whenever you want to test your files you can make a new turing script and run it with sbatch.
