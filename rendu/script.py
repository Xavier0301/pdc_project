import numpy as np
import string
import random
import io
import argparse
import pathlib
import subprocess
import sys

#################################################
#                                               #
#   HELPER FUNCTIONS                            #
#                                               #
#################################################

def stringToBits(string):
    """
    Function to decode a string of utf-8 encoded characters into bits

    Parameters:
        string: the string to decode

    Returns:
        the decoded string as an array
    """
    res = []
    byte_string = string.encode('utf-8')
    for b in byte_string:
        bit_array = bin(b)[2:]
        bit_array = '00000000'[len(bit_array):] + bit_array
        # we drop the first bit because it is always 0 for 1 byte utf-8 encoded chars
        res.extend(bit_array[1:])
    return np.array(res, dtype='int64')

def binaryToString(bits):
    """
    Function to encode a string of bits into a string of utf-8 encoded characters.

    Parameters:
        bits: the bit array to encode
    
    Returns:
        the encoded array as a string
    """
    bits = bits.tolist()
    byte_string = ""
    for char_index in range(len(bits)//7):
        bit_list = bits[char_index*7:(char_index+1)*7]
        byte = chr(int(''.join([str(bit) for bit in bit_list]), 2))
        byte_string += byte
    return byte_string

def genCodebook(initial=[[1]], k=5):
    """
    Function to generate a orthogonal code codebook as in exercise 2 of the graded homework 2.

    Parameters:
        initial: the initialization state, [[1]] by default
        k: parameter defining the dimensionality of the codes which is 2^k

    Returns:
        the codebook as an array
    """

    def inverseSign(word):
        return [-i for i in word]

    if k == 0:
        return initial
    else:
        new = [word + inverseSign(word) for word in initial] + [word + word for word in initial]
        return genCodebook(new, k-1)

def binaryToDecimal(bits):
    """
    Function to decode an array of bits into a decimal number

    Parameters:
        bits: the bit array to decode

    Returns:
        the decoded decimal number as an integer
    """
    res = 0
    for i in range(len(bits)):
        res += bits[i] * (2**i)
    return int(res)

def decimalToBinary(d, s):
    """
    Funtion to encode a decimal number into an array of bits

    Parameters:
        d: the decimal to encode
        s: the number of bits to use

    Returns:
        the encoded decimal number as an array of bits
    """
    res = np.zeros(s, dtype='int64')
    tmp = list(bin(d).replace("0b", ""))[::-1]
    if (len(tmp) > s):
        print("WARNING number too big to fit into expected binary length!")
    res[:len(tmp)] = tmp
    return res

def encode(chan_input, codebook, k):
    """
    Makes a string for transmission.

    Parameters:
        chan_input: the input string
        codebook: the orthogonal codebook used by the transmission
        k: the parameter used to build the codebook

    Returns:
        an array containing the encoded string
    """
    # read the content of the file
    

    # first encode the string
    chan_input = stringToBits(chan_input) 

    # add padding and a block containing the padding size
    n = len(chan_input)
    n_blocks = n // k 
    padding_size = k - n % k
    if (padding_size < 10):
        padding = np.zeros(padding_size).astype('int64')
        chan_input = np.append(chan_input, padding)
        chan_input = np.append(chan_input, decimalToBinary(padding_size, k))
        n_blocks += 2

    # map each block of bits to a code from the codebook
    res = np.array([], dtype='int64')
    
    for i in range(n_blocks):
        res = np.append(res, codebook[binaryToDecimal(chan_input[i*k:(i+1)*k])])

    chan_input = res.flatten()

    return chan_input

def decode(chan_output, codebook, k):
    """
    Decodes the result of a channel transmission.

    Parameters:
        chan_output: array containing the channel output to decode
        codebook: the orthogonal codebook used by the transmission
        k: the parameter used to build the codebook

    Returns:
        the decoded channel output as a string
    """
    # retrieve input codewords
    def find(codebook, codeword):
        res = np.array([])
        for c in codebook:
            res = np.append(res, np.inner(c, codeword))
        return res.argmax()

    chan_output = np.split(chan_output, len(chan_output)/len(codebook[0]))
    chan_output = np.array([find(codebook, codeword) for codeword in chan_output])
    chan_output = np.array([decimalToBinary(codeword, k) for codeword in chan_output]).flatten()

    # remove padding
    padding_size = binaryToDecimal(chan_output[-k:])
    if (padding_size < k):
        chan_output = chan_output[:-(padding_size + k)]

    # transform back into string
    chan_output = binaryToString(chan_output)

    return chan_output

#################################################
#                                               #
#   MAIN                                        #
#                                               #
#################################################

def readyForTransmission(input_filename, output_filename):
    input_string = ""
    with io.open(input_filename, encoding='utf8') as f:
        input_string = f.read()

    print("Making the string\n\t" + input_string + "\nready for transmission...")

    encoded_chan_input = encode(input_string, codebook, k)
    np.savetxt(output_filename, encoded_chan_input)

    print("... ready for transmission! You can check the content in " + output_filename + '\n')

    return input_string

def transmit(input_filename, output_filename, python):
    chan_input = np.loadtxt(input_filename)

    print("Transmitting...")
    print("... n is " + str(len(chan_input)) + " ...")

    command = subprocess.run([python, 'client.py', '--input_file='+input_filename, '--output_file='+output_filename, '--srv_hostname' ,'iscsrv72.epfl.ch', '--srv_port', '80'], capture_output=True)
    sys.stdout.buffer.write(command.stdout)
    sys.stderr.buffer.write(command.stderr)

    print("... received the result! You can check the content in " + output_filename + '\n')

def decodeResult(channel_output_filename, result_filename, input_string):
    print("Decoding...")

    output_string = decode(np.loadtxt(channel_output_filename), codebook, k)
    diff = sum(output_string[i] != input_string[i] for i in range(len(output_string)))
    with io.open(result_filename, "w", encoding='utf8') as f:
        f.write(output_string)

    print("..the input was\n\t" + input_string + "\nand we decoded\n\t" + output_string + "\nafter transmission which means that we made " + str(diff) + " mistakes!")
    print("The resulting string is available in " + result_filename + '\n')

def argument_parser():
    parser = argparse.ArgumentParser(formatter_class= argparse.RawTextHelpFormatter)

    parser.add_argument('--input_file', type=str, required=True, help='.txt contaiing the utf-8 encoded string to transmit.')
    parser.add_argument('--python', type=str, required=False, default='python3', help='name of the prompt to call python from command line on your machine. Is python3 by default.')

    args = parser.parse_args()

    args.input_file = pathlib.Path(args.input_file).resolve(strict=True)
    if not (args.input_file.is_file() and
            (args.input_file.suffix == '.txt')):
        raise ValueError('Parameter[input_file] is not a .txt file.')

    return args

if __name__ == '__main__':
    k = 10
    codebook = genCodebook(k=k)

    result_filename = "result.txt"
    chan_input_filename = "channel_input.txt"
    chan_output_filename = "channel_output.txt"

    args = argument_parser()

    input_string = readyForTransmission(args.input_file, chan_input_filename)
    transmit(chan_input_filename, chan_output_filename, args.python)
    decodeResult(chan_output_filename, result_filename, input_string)