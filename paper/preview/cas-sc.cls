%% PREVIEW STAND-IN, NOT THE ELSEVIER CLASS.
%% Implements just enough of the cas-sc interface (els-cas-templates) to typeset paper/main.tex for proof
%% reading when CTAN is unreachable. For submission, compile main.tex with the official cas-sc.cls
%% (Overleaf "Elsevier CAS" template), which this file must never replace.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cas-sc}[2026/10/12 preview stand-in for Elsevier cas-sc]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[11pt]{article}
\RequirePackage[margin=2.2cm,a4paper]{geometry}
\RequirePackage[T1]{fontenc}
\RequirePackage{mathptmx}
\RequirePackage{array,xcolor,keyval,etoolbox}
\RequirePackage[hidelinks]{hyperref}
\RequirePackage{caption}
\captionsetup{font=small,labelfont=bf,labelsep=period,justification=raggedright,singlelinecheck=false}
\captionsetup[table]{position=top}
\setlength{\mathindent}{2em}

\newcommand{\shorttitle}[1]{\gdef\@shorttitle{#1}}
\newcommand{\shortauthors}[1]{\gdef\@shortauthors{#1}}
\RenewDocumentCommand{\title}{o m}{\gdef\@title{#2}}
\def\@authorlist{}
\RenewDocumentCommand{\author}{o m o}{\g@addto@macro\@authorlist{\@sep#2\textsuperscript{#1}}\gdef\@sep{, }}
\gdef\@sep{}
\newcommand{\cormark}[1][1]{\g@addto@macro\@authorlist{\textsuperscript{*}}}
\def\@emails{}
\newcommand{\ead}[1]{\g@addto@macro\@emails{#1 }}
\def\@affils{}
\define@key{aff}{organization}{\def\aff@org{#1}}
\define@key{aff}{addressline}{\def\aff@addr{#1}}
\define@key{aff}{city}{\def\aff@city{#1}}
\define@key{aff}{postcode}{\def\aff@pc{#1}}
\define@key{aff}{state}{\def\aff@st{#1}}
\define@key{aff}{country}{\def\aff@ctry{#1}}
\NewDocumentCommand{\affiliation}{o m}{\setkeys{aff}{#2}%
  \xdef\@affils{\unexpanded\expandafter{\@affils}\noexpand\par\noexpand\textsuperscript{#1}%
  \unexpanded\expandafter{\aff@org}, \unexpanded\expandafter{\aff@addr}, \unexpanded\expandafter{\aff@city}
  \unexpanded\expandafter{\aff@pc}, \unexpanded\expandafter{\aff@ctry}}}
\def\@cortext{}
\newcommand{\cortext}[2][1]{\gdef\@cortext{#2}}
\newcommand{\sep}{\unskip\ \textperiodcentered\ }
\newbox\abs@box \newbox\hl@box \newbox\kw@box
\RenewDocumentEnvironment{abstract}{+b}{\global\setbox\abs@box\vbox{\hsize\textwidth\small
  \parskip 4pt\noindent\textbf{Abstract}\par #1}}{}
\NewDocumentEnvironment{highlights}{+b}{\global\setbox\hl@box\vbox{\hsize\textwidth\small
  \noindent\textbf{Highlights}\begin{itemize}\setlength\itemsep{0pt}#1\end{itemize}}}{}
\NewDocumentEnvironment{keywords}{+b}{\global\setbox\kw@box\vbox{\hsize\textwidth\small
  \noindent\textbf{Keywords}\quad #1}}{}
\NewDocumentEnvironment{graphicalabstract}{+b}{}{}
\renewcommand{\maketitle}{%
  \begin{flushleft}
  {\color{gray}\footnotesize Preview typeset with a stand-in class. Submission uses Elsevier cas-sc.\par}\medskip
  {\LARGE\bfseries\@title\par}\bigskip
  {\large\@authorlist\par}\smallskip
  {\footnotesize\@affils\par\smallskip *\,\@cortext\ E-mail address\space\@emails\par}
  \end{flushleft}
  \noindent\rule{\textwidth}{0.4pt}\par\medskip
  \noindent\usebox\hl@box\par\medskip
  \noindent\usebox\abs@box\par\medskip
  \noindent\usebox\kw@box\par
  \noindent\rule{\textwidth}{0.4pt}\par\bigskip}

% CAS float interface: [width=..,cols=..,pos=..]
\newlength{\tblwidth}
\define@key{flt}{width}{\setlength{\tblwidth}{#1}}
\define@key{flt}{cols}{}
\define@key{flt}{pos}{\def\flt@pos{#1}}
\define@key{flt}{align}{}
\let\cas@oldtable\table \let\cas@oldendtable\endtable
\let\cas@oldfigure\figure \let\cas@oldendfigure\endfigure
\renewenvironment{table}[1][]{\setlength{\tblwidth}{\linewidth}\def\flt@pos{htbp}\setkeys{flt}{#1}%
  \edef\x{\noexpand\cas@oldtable[\flt@pos]}\x\centering}{\cas@oldendtable}
\renewenvironment{figure}[1][]{\def\flt@pos{htbp}\setkeys{flt}{#1}%
  \edef\x{\noexpand\cas@oldfigure[\flt@pos]}\x}{\cas@oldendfigure}
\newcolumntype{L}{@{\extracolsep{\fill}}>{\raggedright\arraybackslash}l}
\newcolumntype{R}{@{\extracolsep{\fill}}r}
\newcolumntype{C}{@{\extracolsep{\fill}}c}
\endinput
