input_str = "What is the weather of home?"
print(f"""Give back the location from the given sentence in a format that is *wttr.in* compatible.
 
                The rules are as follows:
                ```
                1. Lowercase all locations
                2. If the location is an airport, use the 3 letter code.
                3. If there is a space in-between the location and it is not an airport, replace whitespace with '+'
                4. If the location is not a city, append '~' to the start of the output
                ```
                Examples:
                ```
                Input: What is the weather in Denver?
                Output: denver
                
                Input: What is the weather at Ronald Reagan International airport?
                Output: dca
                
                Input: Weather of New Mexico?
                Output: new+mexico
                
                Input: Give me the weather of DCA.
                Output: dca
                
                Input: What is the weather at the Statute of Liberty?
                Output: ~statue+of+liberty
                
                Input: What is the weather like at the colosseum?
                Output: ~colosseum
                ```
                
                Using these rules, decode this sentence:
                {input_str}?""")