#ifndef VIS_ASTRIX_H
#define VIS_ASTRIX_H

// function prototypes
void dispAstrix(void);
void keybAstrix(unsigned char key, int x, int y);
void resizeAstrix(int w, int h);
int ReadFiles(int startFlag);

#define sign(X)  ((X) >= 0.0 ? (1) : -(1)) 

#endif
