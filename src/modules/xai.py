from utils import clear_screen
from xai.GLCM import analyse_GLCM
from xai.LTEM_Cosine_Similarity import cosine_similarity

def prompt_analysis():
    print("1. GLCM feature Map Analysis")
    print("2. LTEM feature Map Cosine Similarity Analysis")
    print("3. Exit")

    choice = None
    while True:
            try:
                choice = int(input("Select Function (1-3): "))
                if 1 <= choice <= 3:
                    break  # Exit the loop if the input is valid
                else:
                    print("Please choose one of the 3 available functions.")
            except ValueError:
                print("That's not an integer. Please try again.")

    return choice

def prompt_model():
    print("1. DeeplapV3")
    print("2. FCN")
    print("3. U-Net")

    choice = None
    while True:
            try:
                choice = int(input("Select Model (1-3): "))
                if 1 <= choice <= 3:
                    break  # Exit the loop if the input is valid
                else:
                    print("Please choose one of the 3 available functions.")
            except ValueError:
                print("That's not an integer. Please try again.")

    return choice

def xai_module():
    clear_screen() #clears the terminal screen
    print("Welcome to the Texture Analysis Module")

    choice_analysis = prompt_analysis()

    #perform user requested tasks
    while choice_analysis != 3:
        
        if choice_analysis == 1:
            analyse_GLCM(prompt_model())

        else:
            cosine_similarity()
            
        choice_analysis = prompt_analysis(prompt_model())

    return