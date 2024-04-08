/*
*  _   _ _   _  ____    _    ____  
* | \ | | | | |/ ___|  / \  |  _ \ 
* |  \| | | | | |     / _ \ | |_) |
* | |\  | |_| | |___ / ___ \|  _ < 
* |_| \_|\___/ \____/_/   \_\_| \_\
* 
*/
// Created by Kaustubh Shivdikar
//
// (C) All Rights Reserved

#ifndef DEBUGGER_H
#define DEBUGGER_H

#include <iostream>
#include "hip/hip_runtime.h"

// Enum used to select formatting option
enum color{
	black, bold_grey, red, bold_red, green, bold_green, yellow, bold_yellow, blue, bold_blue, purple, bold_purple, cyan, bold_cyan, white, bold_white, end
};

// Static array to translate the options
static const char* format[] = {

	"\e[0;30m", // Black
	"\e[1;30m", // Bold Grey
	"\e[0;31m", // Red
	"\e[1;31m", // Bold Red
	"\e[0;32m", // Green
	"\e[1;32m", // Bold Green
	"\e[0;33m", // Yellow
	"\e[1;33m", // Bold Yellow
	"\e[0;34m", // Blue
	"\e[1;34m", // Bold Blue
	"\e[0;35m", // Purple
	"\e[1;35m", // Bold Purple
	"\e[0;36m", // Cyan
	"\e[1;36m", // Bold Cyan
	"\e[0;37m", // White
	"\e[1;37m", // Bold White
	"\e[0m" // Color reset
};

// Helper class to easily change the color of text
class formatted_message
{
	public:
		std::string _msg;
		color _opt;

		formatted_message(color opt, std::string msg): _msg(msg), _opt(opt){};
		friend std::ostream& operator<<(std::ostream& os, const formatted_message& fm);
};

// Cuda errcheck helper function
#define gpuErrchk(ans)                        \
	{                                         \
		gpuAssert((ans), __FILE__, __LINE__); \
	}

inline void gpuAssert(hipError_t code, const char *file, int line, bool abort = true)
{
	if (code != hipSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
		if (abort)
			exit(code);

	}
}

#endif // DEBUGGER_H
