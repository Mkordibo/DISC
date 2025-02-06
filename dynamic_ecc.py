
#################################################
# This class implements the Dynamic ECC of      #
# Section 5 in the paper                        #
#################################################

class DynamicECC:
    """
    This class deals with dynamic error correction code introduced  in the paper
    
    every object has the following properties,
    input: input bitstreams that are coded 
    stream: list of symbols (codeword)
    
    """
    def __init__(self, input):
        self.input = input
        self.last_index_written = -1
        self.suffix_to_remove = 0
        self.stream = []
        self.default_symbol = "0"

    @staticmethod
    def decode(stream):
        """
        This function decodes a codeword to a bit stream

        Parameters
        ----------
        stream : list(str)
            codeword, list of symbols.

        Returns
        -------
        str
            list of input bits (0's and 1's).

        """
        message = []
        for symbol in stream:
            if symbol == "<":
                if message:
                    message.pop()
            else:
                message.append(symbol)

        return "".join(message)

    def update(self, symbol):
        """
        This function updates the ECC, given the symbol that decoder will decide 
        for the current chunk, and the inout payload

        Parameters
        ----------
        symbol : str
            The symbol that will be decided for the current chunk of tokens.

        Returns
        -------
        None.

        """
        last_index_written = self.last_index_written
        suffix_to_remove = self.suffix_to_remove

        self.stream.append(symbol)

        # There is an incorrect suffix, 
        if suffix_to_remove: # in this case the decoder correctly decides on symbol '<', 
        # which removes one of the previously incorrect sent symbols
            if symbol == "<":
                self.suffix_to_remove = suffix_to_remove - 1
            else: # in this case the decoder incorrectly decides on symbol '0' or '1', 
            # therefore, adding one extra incorrect symbol to the code word
                self.suffix_to_remove = suffix_to_remove + 1
            return None

        # The previous stream is correct
        next_symbol = self.input[last_index_written+1] if (last_index_written + 1 < len(self.input)) \
            else self.default_symbol

        if symbol == next_symbol: # The symbol that the decoder will decide is 
        # equal to the actual payload bit, which means, the next symbol can be the
        # bit of the payload, otherwise the next symbol has to be '<'. For example, 
        # the bit in payload is '0' and the symbol that the decoder will decide 
        # is also '0', in this case 'last_index_written' will increase by 1
            self.last_index_written = last_index_written + 1
        elif symbol == "<": # The symbol that decoder has decided is '<', which means 
        # the last payload that was sent before will not be considered by the 
        # decoder and has to be transfered again in the code word
            if last_index_written > -1:
                self.last_index_written = last_index_written - 1
        else: # The symbol decided by the decoder is '0' or '1', but it is wrong 
        # compared to hte payload (the payload can be '0' and the symbol being '1'),
        # in this case the next symbol should remove this incorrect symbol
            self.suffix_to_remove = 1
        return None

    def next_symbol(self):
        """
        This function generates the next symbol in the code word

        Returns
        -------
        str
            The next symbol of the .

        """
        if self.suffix_to_remove:
            return "<"
        else:
            next_symbol = self.input[self.last_index_written + 1] if (self.last_index_written + 1 < len(self.input)) \
                else self.default_symbol
            return next_symbol
