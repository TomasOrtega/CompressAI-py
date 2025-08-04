import struct
import numpy as np
from typing import List

# rANS constants
RANS64_L = 1 << 31  # Lower bound of our normalization interval
RANS_BYTE_L = 1 << 23  # For byte-based rANS, but we're using 32-bit chunks

class BufferedRansEncoder:
    """Pure Python implementation of BufferedRansEncoder using rANS algorithm"""
    
    def __init__(self):
        self._symbols = []
        self.precision = 16
        self.bypass_precision = 4
        self.max_bypass_val = (1 << self.bypass_precision) - 1
    
    def encode_with_indexes(self, symbols, indexes, cdfs, cdf_lengths, offsets):
        """Encode symbols using rANS with given CDFs and indexes"""
        assert len(symbols) == len(indexes)
        
        # Store symbols in reverse order (we encode backwards in ANS)
        for i in range(len(symbols)):
            cdf_idx = indexes[i]
            assert 0 <= cdf_idx < len(cdfs)
            
            cdf = cdfs[cdf_idx]
            cdf_length = cdf_lengths[cdf_idx]
            offset = offsets[cdf_idx]
            max_value = cdf_length - 2
            
            # Adjust symbol by offset
            value = symbols[i] - offset
            
            # Handle out-of-range values with bypass coding
            raw_val = 0
            if value < 0:
                raw_val = -2 * value - 1
                value = max_value
            elif value >= max_value:
                raw_val = 2 * (value - max_value)
                value = max_value
            
            assert 0 <= value < cdf_length - 1
            
            # Store the main symbol
            freq_start = cdf[value]
            freq_range = cdf[value + 1] - freq_start
            self._symbols.append({
                'start': freq_start,
                'range': freq_range,
                'bypass': False
            })
            
            # Handle bypass coding if needed
            if value == max_value:
                # Encode the raw value using bypass coding
                n_bypass = 0
                temp_val = raw_val
                while temp_val > 0:
                    n_bypass += 1
                    temp_val >>= self.bypass_precision
                
                # Encode number of bypasses
                val = n_bypass
                while val >= self.max_bypass_val:
                    self._symbols.append({
                        'start': self.max_bypass_val,
                        'range': 1,
                        'bypass': True
                    })
                    val -= self.max_bypass_val
                
                self._symbols.append({
                    'start': val,
                    'range': 1,
                    'bypass': True
                })
                
                # Encode raw value in chunks
                for j in range(n_bypass):
                    chunk_val = (raw_val >> (j * self.bypass_precision)) & self.max_bypass_val
                    self._symbols.append({
                        'start': chunk_val,
                        'range': 1,
                        'bypass': True
                    })
    
    def flush(self) -> bytes:
        """Flush the encoder and return the compressed bitstream"""
        # Initialize rANS state
        rans_state = RANS64_L
        output_buffer = []
        
        # Process symbols in reverse order (ANS encodes backwards)
        for sym in reversed(self._symbols):
            if not sym['bypass']:
                # Regular rANS encoding
                rans_state = self._rans_enc_put(
                    rans_state, output_buffer, 
                    sym['start'], sym['range'], self.precision
                )
            else:
                # Bypass mode - encode raw bits
                rans_state = self._rans_enc_put_bits(
                    rans_state, output_buffer,
                    sym['start'], self.bypass_precision
                )
        
        # Flush final state
        self._rans_enc_flush(rans_state, output_buffer)
        
        # Convert output buffer to bytes
        return b''.join(struct.pack('<I', x) for x in reversed(output_buffer))
    
    def _rans_enc_put(self, state, output_buffer, start, freq, precision):
        """Encode a symbol using rANS"""
        # Renormalization
        x_max = ((RANS64_L >> precision) << 32) * freq
        if state >= x_max:
            output_buffer.append(state & 0xFFFFFFFF)
            state >>= 32
        
        # State update: x = C(s,x) = (x/freq)*M + bias + (x%freq)
        # For rANS: C(s,x) = (x/freq) << precision + start + (x%freq)
        state = ((state // freq) << precision) + (state % freq) + start
        return state
    
    def _rans_enc_put_bits(self, state, output_buffer, val, nbits):
        """Encode raw bits for bypass mode"""
        assert nbits <= 16
        assert val < (1 << nbits)
        
        # Renormalization
        freq = 1 << (16 - nbits)
        x_max = ((RANS64_L >> 16) << 32) * freq
        if state >= x_max:
            output_buffer.append(state & 0xFFFFFFFF)
            state >>= 32
        
        # State update for raw bits
        state = (state << nbits) | val
        return state
    
    def _rans_enc_flush(self, state, output_buffer):
        """Flush the final rANS state"""
        output_buffer.append(state & 0xFFFFFFFF)
        output_buffer.append((state >> 32) & 0xFFFFFFFF)


class RansEncoder:
    """Pure Python implementation of RansEncoder"""
    
    def encode_with_indexes(self, symbols, indexes, cdfs, cdf_lengths, offsets):
        encoder = BufferedRansEncoder()
        encoder.encode_with_indexes(symbols, indexes, cdfs, cdf_lengths, offsets)
        return encoder.flush()


class RansDecoder:
    """Pure Python implementation of RansDecoder"""
    
    def __init__(self):
        self._stream = b''
        self._ptr = 0
        self._rans_state = 0
        self.precision = 16
        self.bypass_precision = 4
        self.max_bypass_val = (1 << self.bypass_precision) - 1
    
    def set_stream(self, stream: bytes):
        """Set the compressed bitstream"""
        self._stream = stream
        self._ptr = 0
        # Initialize rANS state
        self._rans_dec_init()
    
    def decode_with_indexes(self, stream, indexes, cdfs, cdf_lengths, offsets):
        """Decode symbols from stream"""
        self.set_stream(stream)
        return self.decode_stream(indexes, cdfs, cdf_lengths, offsets)
    
    def decode_stream(self, indexes, cdfs, cdf_lengths, offsets):
        """Decode symbols using the current stream"""
        output = []
        
        for idx in indexes:
            cdf_idx = idx
            assert 0 <= cdf_idx < len(cdfs)
            
            cdf = cdfs[cdf_idx]
            cdf_length = cdf_lengths[cdf_idx]
            offset = offsets[cdf_idx]
            max_value = cdf_length - 2
            
            # Decode symbol
            cum_freq = self._rans_dec_get(self.precision)
            
            # Binary search in CDF
            symbol = 0
            for i in range(cdf_length - 1):
                if cdf[i] <= cum_freq < cdf[i + 1]:
                    symbol = i
                    break
            
            # Advance decoder state
            self._rans_dec_advance(cdf[symbol], cdf[symbol + 1] - cdf[symbol], self.precision)
            
            value = symbol
            
            # Handle bypass decoding if at max_value
            if value == max_value:
                # Decode number of bypasses
                n_bypass = 0
                while True:
                    val = self._rans_dec_get_bits(self.bypass_precision)
                    n_bypass += val
                    if val < self.max_bypass_val:
                        break
                
                # Decode raw value
                raw_val = 0
                for j in range(n_bypass):
                    val = self._rans_dec_get_bits(self.bypass_precision)
                    raw_val |= val << (j * self.bypass_precision)
                
                # Reconstruct original value
                if raw_val & 1:
                    value = -(raw_val >> 1) - 1
                else:
                    value = (raw_val >> 1) + max_value
            
            output.append(value + offset)
        
        return output
    
    def _rans_dec_init(self):
        """Initialize decoder state from stream"""
        if len(self._stream) < 8:
            raise ValueError("Stream too short")
        
        # Read initial state (64-bit)
        data = struct.unpack('<II', self._stream[:8])
        self._rans_state = (data[1] << 32) | data[0]
        self._ptr = 8
    
    def _rans_dec_get(self, precision):
        """Get cumulative frequency for current state"""
        return self._rans_state & ((1 << precision) - 1)
    
    def _rans_dec_advance(self, start, freq, precision):
        """Advance decoder state after decoding a symbol"""
        # Remove encoded symbol from state
        self._rans_state = freq * (self._rans_state >> precision) + (self._rans_state & ((1 << precision) - 1)) - start
        
        # Renormalize if needed
        if self._rans_state < RANS64_L:
            if self._ptr < len(self._stream):
                new_data = struct.unpack('<I', self._stream[self._ptr:self._ptr+4])[0]
                self._rans_state = (self._rans_state << 32) | new_data
                self._ptr += 4
    
    def _rans_dec_get_bits(self, nbits):
        """Decode raw bits for bypass mode"""
        val = self._rans_state & ((1 << nbits) - 1)
        
        # Update state
        self._rans_state >>= nbits
        
        # Renormalize if needed
        if self._rans_state < RANS64_L:
            if self._ptr < len(self._stream):
                new_data = struct.unpack('<I', self._stream[self._ptr:self._ptr+4])[0]
                self._rans_state = (self._rans_state << 32) | new_data
                self._ptr += 4
        
        return val