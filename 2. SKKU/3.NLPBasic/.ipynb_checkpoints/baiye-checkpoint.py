{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import math, sys\n",
    "from konlpy.tag import Twitter\n",
    "class BayesianFilter:\n",
    "    def __init__(self):\n",
    "        self.words = set() # 출현한 단어 기록\n",
    "        self.word_dict = {} # 카테고리마다의 출현 횟수 기록\n",
    "        self.category_dict = {} # 카테고리 출현 횟수 기록\n",
    "    # 형태소 분석하기 --- (※1)\n",
    "    def split(self, text):\n",
    "        results = []\n",
    "        twitter = Twitter()\n",
    "        # 단어의 기본형 사용\n",
    "        malist = twitter.pos(text, norm=True, stem=True)\n",
    "        for word in malist:\n",
    "            # 어미/조사/구두점 등은 대상에서 제외 \n",
    "            if not word[1] in [\"Josa\", \"Eomi\", \"Punctuation\"]:\n",
    "                results.append(word[0])\n",
    "        return results\n",
    "    # 단어와 카테고리의 출현 횟수 세기 --- (※2)\n",
    "    def inc_word(self, word, category):\n",
    "        # 단어를 카테고리에 추가하기\n",
    "        if not category in self.word_dict:\n",
    "            self.word_dict[category] = {}\n",
    "        if not word in self.word_dict[category]:\n",
    "            self.word_dict[category][word] = 0\n",
    "        self.word_dict[category][word] += 1\n",
    "        self.words.add(word)\n",
    "    def inc_category(self, category):\n",
    "        # 카테고리 계산하기\n",
    "        if not category in self.category_dict:\n",
    "            self.category_dict[category] = 0\n",
    "        self.category_dict[category] += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
