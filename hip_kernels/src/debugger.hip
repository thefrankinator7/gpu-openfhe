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

#include "debugger.hpp"

std::ostream& operator<<(std::ostream& os, const formatted_message& fm){
	return os << format[fm._opt] << fm._msg << format[color::end] ;
}