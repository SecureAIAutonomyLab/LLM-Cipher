def calculate_evasion_rate(recall, total_machine_generated_samples=200):
    TP = recall * total_machine_generated_samples
    FN = total_machine_generated_samples - TP
    evasion_rate = FN / total_machine_generated_samples
    return evasion_rate

if __name__ == "__main__":
    while True:
        try:
            # Get input from user
            recall = float(input("Please enter the recall value (or type 'exit' to quit): "))
            
            # Calculate and print evasion rate
            evasion_rate = calculate_evasion_rate(recall)
            print(f"The evasion rate for recall {recall} is: {evasion_rate}\n")
        
        except ValueError as e:
            # Check if user wants to exit
            if str(e).lower() == "could not convert string to float: 'exit'":
                break
            print("Invalid input. Please enter a valid number for recall or type 'exit' to quit.")
