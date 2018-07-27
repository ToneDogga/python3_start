// dynstrlb.cpp - dynamic string class
#include <iostream.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <new.h>    // this library will assume the new handler is set.

#include "dynstrlb.h"

int search(char *searchstr, char *word);

// *** NO-ARG CONSTRUCTOR
// **********************
String::String()
{
	_str = NULL;
	_len = 0;
}

// *** ONE-ARG CONSTRUCTOR
// ***********************
String::String(char *instring)
{

	if (!instring)
	{
		_len = 0;
		_str = 0;
		return;
	}

	_len = strlen(instring);
	_str = new char[_len + 1];			
	strcpy(_str,instring);
}
		
// *** DESTRUCTOR
// **************
String::~String()
{
	delete _str;
}
		
// anytime you use dynamic memory in a class you MUST
// provide a copy constructor and overload the assignment
// operator
		
// *** COPY CONSTRUCTOR
// ********************
String::String(String &other)
{
	_len = other._len;
	_str = new char[_len + 1];
	strcpy(_str,other._str);
}
		
// *** OVERLOAD ASSIGNMENT
// ***********************
String &String::operator = (String &other)
{
	if (&other == this)
		return *this;
		
	if (_str) delete _str;
	_len = other._len;
	_str = new char[_len + 1];
	strcpy(_str,other._str);
	return *this;
}

// *** OVERLOAD ASSIGNMENT
// ***********************
String &String::operator = (const char *instr)
{
	if (_str) delete _str;
	_len = strlen(instr);
	_str = new char[_len + 1];
	strcpy(_str,instr);
	return *this;
}
	
// *** OVERLOAD PLUS
// *****************
String String::operator + (String& ss)
{
	String Temp;
			
	Temp._len = _len + ss._len;
	Temp._str = new char[_len + ss._len + 1];
			
	strcpy(Temp._str,_str);
	strcat(Temp._str,ss._str);
	return Temp;
}

// *** OVERLOAD PLUS
// *****************
String String::operator + (char *ss)
{
	String Temp;
	int slen = strlen(ss);
			
	Temp._len = _len + slen;
	Temp._str = new char[_len + slen + 1];
			
	strcpy(Temp._str,_str);
	strcat(Temp._str,ss);
	return Temp;
}

// *** CONVERSION OPERATOR
// ***********************		
String::operator char *()
{
	return(_str);
}

// *** OVERLOAD MINUS
// ******************		
String String::operator - (String& ss)
{
	String Temp;
	int start_index=0,i=0,j=0;
			
	if (ss._len >= _len)
	{
		cout << "String to subtract must be less than object!\n";
		return String(_str);
	}
			
	Temp._len = _len - ss._len;
	Temp._str = new char[(_len - ss._len) + 1];
	start_index = search(_str,ss._str);
			
	if (start_index < 0)
	{
		cout << "String to subtract not in object!\n";
		return String(_str);
	}
			
	for (i=0; i < start_index; i++)
		Temp._str[i] = _str[i];
	
	j=i;		
	if (i + ss._len < _len)
	{	
		i += ss._len;
		while (_str[i])
		{
			Temp._str[j] = _str[i];
			i++; j++;
		}
	}
			
	Temp._str[Temp._len] = '\0';
			
	return Temp;
}

// *** OVERLOAD MINUS
// ******************		
String String::operator - (char *ss)
{
	String Temp;
	int start_index=0,i=0,j=0;
	int slen = strlen(ss);
			
	if (slen >= _len)
	{
		cout << "String to subtract must be less than object!\n";
		return String(_str);
	}
			
	Temp._len = _len - slen;
	Temp._str = new char[(_len - slen) + 1];
	start_index = search(_str,ss);
			
	if (start_index < 0)
	{
		cout << "String to subtract not in object!\n";
		return String(_str);
	}
			
	for (i=0; i < start_index; i++)
		Temp._str[i] = _str[i];
			
	j=i;		
	if (i + slen < _len)
	{	
		i += slen;
		while (_str[i])
		{
			Temp._str[j] = _str[i];
			i++; j++;
		}
	}
			
	Temp._str[Temp._len] = '\0';
			
	return Temp;
}

// *** OVERLOAD COMPARISON 
// ***********************	
short String::operator == (String &ss)
{
	return((strcmp(_str,ss._str) == 0) ? 1 : 0);
}

short String::operator == (char *ss)
{
	return((strcmp(_str,ss) == 0) ? 1 : 0);
}

short String::operator != (String &ss)
{
	return((strcmp(_str,ss._str) == 0) ? 0 : 1);
}

short String::operator != (char *ss)
{
	return((strcmp(_str,ss) == 0) ? 0 : 1);
}
		
short String::operator > (String ss)
{
	return((strcmp(_str,ss._str) > 0) ? 1 : 0);
}

short String::operator < (String ss)
{
	return((strcmp(_str,ss._str) < 0) ? 1 : 0);
}

int search(char *searchstr, char *word)
{
	int idx,i,j=0;
	
	for (i=0; searchstr[i]; i++)
	{
		if (searchstr[i] == word[j])
		{
			if (!j) idx = i;
			j++;
		}
		else
			j = 0;
		
		if (word[j] == '\0')
			break;
	}

	if (word[j] != '\0')
		return(-1);
	else
		return(idx);
}

ostream &operator << (ostream &stream, String &obj)
{
	stream << obj._str;
	return stream;
}

istream &operator >> (istream &stream, String &obj)
{
	char tmp[256];
	int tlen=0;
	stream >> tmp;
	tlen = strlen(tmp);
	obj._str = new char[tlen + 1];
	strcpy(obj._str,tmp);
	obj._len = tlen;
	return stream;
}

// *** DUPLICATE
// *************
char *String::Strdup()
{
	char *outstr=0;
	
	if (!_str)
		return(0);
	 
	outstr = new char[_len + 1];	
	strcpy(outstr,_str);
	
	return(outstr);	
}

// *** OVERLOAD BRACKETS
// *********************
char &String::operator [] (int index)
{
	assert(index <= _len);
	assert(index >= 0);
	return(_str[index]);
}

// *** SUB STRING
// **************
String String::Substr(int start, int numchars)
{
	char *p;
	
	int cnt=0, slen=0;
	
	slen = strlen(_str);
	if ( (slen < 2) || /* is this a valid string? */
	     (start< 1) || /* is start valid? */
	     (start>slen) ) /* is the substring in the string? */
	     return(NULL);
	
	p = new char[numchars + 1];
	start--; /* subtract one since C strings start at 0. */
	while (cnt < numchars)
	{
		if ((_str[start+cnt] == '\0') ||
		    (_str[start+cnt] == '\n') ) break;
			p[cnt] = _str[start+cnt];
		++cnt;
	} /* end of while */
	
	p[cnt] = '\0';
	return String(p);
}

// *** getline
// ***********
void String::getline(istream &stream)
{
	char temp[256];
	int thelen;
	stream.getline(temp,255);
	thelen = strlen(temp);
	
	if (_str) delete _str;
	_str = new char[thelen + 1];
	strcpy(_str,temp);
	_len = thelen;
}

// *** getlen
// **********
int String::getlen()
{
	return _len;
}

int String::Strindex(char *word)
{
	int idx=0;
	
	idx = search(_str,word);
	return idx;
}
