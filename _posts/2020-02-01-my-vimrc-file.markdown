---

title: "My .vimrc file"
layout: post
date: 2020-01-29 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:

- machine learning
- math
star: false
category: blog
author: yotam
description: My .vimrc file
---

# My .vimrc file

```bash
nnoremap ' ;
nnoremap ; l
nnoremap k j
nnoremap j k
nnoremap l h

map <C-o> :NERDTreeToggle<CR>

set rnu
:imap jk <Esc>

colorscheme delek
set clipboard=unnamedplus
set hlsearch
set incsearch
set ignorecase
set smartcase
set so=5
syntax on
let mapleader = "_"

set nocompatible
filetype off
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()

Plugin 'VundleVim/Vundle.vim'
Plugin 'scrooloose/nerdtree'
Plugin 'chrisbra/csv.vim'
Plugin 'joshdick/onedark.vim'

call vundle#end()
filetype plugin indent on
```

