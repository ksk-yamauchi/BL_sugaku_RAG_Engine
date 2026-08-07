ベーシックレベル数学Ⅰ 第11講 PART4

# PART4 三角比の相互関係

## 要点① 三角比の相互関係

右の図の直角三角形において
$a = c \sin \theta$
$b = c \cos \theta$
である。よって
$$\tan \theta = \frac{a}{b} = \frac{c \sin \theta}{c \cos \theta} = \frac{\sin \theta}{\cos \theta} \quad \cdots\cdots ①$$

[図の説明: 角Cが直角の直角三角形ABC。斜辺AB=c、底辺AC=b、対辺BC=a、角A=θと示されている。]

また，三平方の定理により $a^2 + b^2 = c^2$ であるから
$$(c \sin \theta)^2 + (c \cos \theta)^2 = c^2$$
$$c^2 (\sin \theta)^2 + c^2 (\cos \theta)^2 = c^2$$
両辺を $c^2$ で割って $(\sin \theta)^2 + (\cos \theta)^2 = 1 \quad \cdots\cdots ②$$

さらにこの等式の両辺を $(\cos \theta)^2$ で割ると
$$\left(\frac{\sin \theta}{\cos \theta}\right)^2 + 1^2 = \frac{1}{(\cos \theta)^2} \quad \text{すなわち} \quad (\tan \theta)^2 + 1 = \frac{1}{(\cos \theta)^2} \quad \cdots\cdots ③$$

$(\sin \theta)^2$, $(\cos \theta)^2$, $(\tan \theta)^2$ を，それぞれ $\sin^2 \theta$, $\cos^2 \theta$, $\tan^2 \theta$ と表すと，①，②，③より，次の公式が成り立つ。

---

### 三角比の相互関係

1. $\tan \theta = \frac{\sin \theta}{\cos \theta}$
2. $\sin^2 \theta + \cos^2 \theta = 1$
3. $1 + \tan^2 \theta = \frac{1}{\cos^2 \theta}$

---

三角比の相互関係を用いると，$\theta$ が鋭角のとき，$\sin \theta$, $\cos \theta$, $\tan \theta$ のうちどれか 1 つの値から，他の 2 つの値を求めることができる。

#### 例
$\theta$ が鋭角で，$\sin \theta = \frac{2}{3}$ のとき，$\cos \theta$, $\tan \theta$ の値を求める。

$\sin^2 \theta + \cos^2 \theta = 1$ であるから
$$\cos^2 \theta = 1 - \sin^2 \theta = 1 - \left(\frac{2}{3}\right)^2 = 1 - \frac{4}{9} = \frac{5}{9}$$

$\cos \theta > 0$ であるから
$$\cos \theta = \sqrt{\frac{5}{9}} = \frac{\sqrt{5}}{3}$$

また
$$\tan \theta = \frac{\sin \theta}{\cos \theta} = \frac{2}{3} \div \frac{\sqrt{5}}{3} = \frac{2}{3} \times \frac{3}{\sqrt{5}} = \frac{2}{\sqrt{5}}$$

※ $\theta$ が鋭角で，$\sin \theta = \frac{2}{3}$ のとき，右図において三平方の定理により
$$AC = \sqrt{3^2 - 2^2} = \sqrt{5}$$
よって $\cos \theta = \frac{\sqrt{5}}{3}$, $\tan \theta = \frac{2}{\sqrt{5}}$

[図の説明: 角Cが直角の直角三角形ABC。斜辺AB=3、高さBC=2、角A=θと示されている。]

---

### 問題 4
(1) $\theta$ が鋭角で，$\cos \theta = \frac{1}{4}$ のとき，$\sin \theta$, $\tan \theta$ の値を求めよ。

(2) $\theta$ が鋭角で，$\tan \theta = \frac{\sqrt{5}}{2}$ のとき，$\cos \theta$, $\sin \theta$ の値を求めよ。

---

## Point Pickup

### 三角比の相互関係
1. $\tan \theta = \frac{\sin \theta}{\cos \theta}$
2. $\sin^2 \theta + \cos^2 \theta = 1$
3. $1 + \tan^2 \theta = \frac{1}{\cos^2 \theta}$

$\theta$ が鋭角のとき，この関係を用いることにより，$\sin \theta$, $\cos \theta$, $\tan \theta$ のうちどれか 1 つの値がわかると，他の 2 つの値を求めることができる。

---

# 第11講 PART4 確認問題

## 1
$\theta$ が鋭角で，$\cos \theta = \frac{3}{5}$ のとき，
$$\sin \theta = \frac{[\text{ア}]}{[\text{イ}]}, \quad \tan \theta = \frac{[\text{ウ}]}{[\text{エ}]}$$
である。

## 2
$\theta$ が鋭角で，$\tan \theta = \sqrt{2}$ のとき，
$$\sin \theta = \frac{\sqrt{[\text{オ}]}}{[\text{カ}]}, \quad \cos \theta = \frac{[\text{キ}]}{\sqrt{[\text{ク}]}}$$
である。

---

# 解答・解説

## PART4 三角比の相互関係

### 問題 4 解答

#### (1)
$\sin^2 \theta + \cos^2 \theta = 1$ であるから
$$\sin^2 \theta = 1 - \cos^2 \theta = 1 - \left(\frac{1}{4}\right)^2 = 1 - \frac{1}{16} = \frac{15}{16}$$
$\theta$ は鋭角より，$\sin \theta > 0$ であるから
$$\sin \theta = \sqrt{\frac{15}{16}} = \frac{\sqrt{15}}{4}$$
また
$$\tan \theta = \frac{\sin \theta}{\cos \theta} = \frac{\sqrt{15}}{4} \div \frac{1}{4} = \frac{\sqrt{15}}{4} \times 4 = \sqrt{15}$$

#### (2)
$1 + \tan^2 \theta = \frac{1}{\cos^2 \theta}$ であるから
$$\frac{1}{\cos^2 \theta} = 1 + \tan^2 \theta = 1 + \left(\frac{\sqrt{5}}{2}\right)^2 = \frac{9}{4}$$
よって
$$\cos^2 \theta = \frac{4}{9}$$
$\theta$ は鋭角より，$\cos \theta > 0$ であるから
$$\cos \theta = \frac{2}{3}$$
また $\tan \theta = \frac{\sin \theta}{\cos \theta}$ であるから
$$\sin \theta = \tan \theta \cos \theta = \frac{\sqrt{5}}{2} \cdot \frac{2}{3} = \frac{\sqrt{5}}{3}$$

---

## 第11講 PART4 確認問題 解答

### 1
**［正解］** ア：4，イ：5，ウ：4，エ：3

**［解説］**
$\sin^2 \theta + \cos^2 \theta = 1$ であるから，［A］
$$\sin^2 \theta = 1 - \cos^2 \theta = 1 - \left(\frac{3}{5}\right)^2 = \frac{16}{25}$$
$\theta$ は鋭角より，$\sin \theta > 0$ であるから，
$$\sin \theta = \sqrt{\frac{16}{25}} = \frac{4}{5}$$
また，$\tan \theta = \frac{\sin \theta}{\cos \theta}$ ［B］
$$\tan \theta = \frac{4}{5} \div \frac{3}{5} = \frac{4}{3}$$

* ［A］ 三角比の相互関係 $\sin^2 \theta + \cos^2 \theta = 1$
* ［B］ 三角比の相互関係 $\tan \theta = \frac{\sin \theta}{\cos \theta}$

---

### 2
**［正解］** オ：6，カ：3，キ：1，ク：3

**［解説］**
$1 + \tan^2 \theta = \frac{1}{\cos^2 \theta}$ であるから，［C］
$$\frac{1}{\cos^2 \theta} = 1 + \tan^2 \theta = 1 + (\sqrt{2})^2 = 3$$
$$\cos^2 \theta = \frac{1}{3}$$
$\theta$ は鋭角より，$\cos \theta > 0$ であるから，
$$\cos \theta = \sqrt{\frac{1}{3}} = \frac{1}{\sqrt{3}}$$
また，$\tan \theta = \frac{\sin \theta}{\cos \theta}$ であるから，［D］
$$\sin \theta = \tan \theta \cos \theta = \sqrt{2} \times \frac{\sqrt{3}}{3} = \frac{\sqrt{6}}{3}$$

* ［C］ 三角比の相互関係 $1 + \tan^2 \theta = \frac{1}{\cos^2 \theta}$
* ［D］ $\tan \theta = \frac{\sin \theta}{\cos \theta}$ から $\sin \theta$ を求める。