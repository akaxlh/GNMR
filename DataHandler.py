import pickle
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from Utils.TimeLogger import log
from Params import args

if args.data == 'yelp':
	predir = 'D:/Datasets/yelp/%s/' % args.target
	behs = ['tip', 'neg', 'neutral', 'pos']
elif args.data == 'ml10m':
	predir = 'D:/Datasets/MultiInt-ML10M/%s/' % args.target
	behs = ['neg', 'neutral', 'pos']
elif args.data == 'ECommerce':
	if args.target == 'click':
		predir = 'D:/Datasets/Tmall/backup/hr_ndcg_click/'
	elif args.target == 'buy':
		predir = 'D:/Datasets/Tmall/backup/hr_ndcg_buy/'
	behs = ['pv', 'fav', 'cart', 'buy']
trnfile = predir + 'trn_'
tstfile = predir + 'tst_'

def loadPretrnEmbeds():
	with open('pretrain/PreTrnEmbed_usr', 'rb') as fs:
		usrEmbeds = pickle.load(fs)
	with open('pretrain/PreTrnEmbed_itm', 'rb') as fs:
		itmEmbeds = pickle.load(fs)
	return np.array(usrEmbeds), np.array(itmEmbeds)

def LoadData(itemBased=False):
	trnMats = list()
	for i in range(len(behs)):
		beh = behs[i]
		path = trnfile + beh
		with open(path, 'rb') as fs:
			mat = (pickle.load(fs) != 0).astype(np.float32)
		trnMats.append(mat)
		if args.target == 'click':
			trnLabel = (mat if i==0 else 1 * (trnLabel + mat != 0))
		elif args.target == 'buy' and i == len(behs) - 1:
			trnLabel = 1 * (mat != 0)
	path = tstfile + 'int'
	with open(path, 'rb') as fs:
		tstInt = np.array(pickle.load(fs))

	# item based
	if itemBased:
		for i in range(len(behs)):
			trnMats[i] = transpose(trnMats[i])
		trnLabel = transpose(trnLabel)
		temTstInt = [None] * trnLabel.shape[0]
		for i in range(trnLabel.shape[1]):
			if tstInt[i] != None:
				temu = i
				temi = tstInt[i]
				temTstInt[temi] = temu
		tstInt = np.array(temTstInt)

	tstStat = (tstInt != None)
	tstUsrs = np.reshape(np.argwhere(tstStat!= False), [-1])
	return trnMats, tstInt, trnLabel, tstUsrs, len(behs)

def prepareGlobalData(trnMats, trnLabel):
	global adjs
	global adj
	global tpadj
	global adjNorm
	global tpadjNorm
	adjs = trnMats
	# adj = (np.sum(trnMats)!=0).astype(np.float32)
	adj = trnLabel.astype(np.float32)
	tpadj = transpose(adj)
	adjNorm = np.reshape(np.array(np.sum(adj, axis=1)), [-1])
	tpadjNorm = np.reshape(np.array(np.sum(tpadj, axis=1)), [-1])
	for i in range(adj.shape[0]):
		for j in range(adj.indptr[i], adj.indptr[i+1]):
			adj.data[j] /= adjNorm[i]
	for i in range(tpadj.shape[0]):
		for j in range(tpadj.indptr[i], tpadj.indptr[i+1]):
			tpadj.data[j] /= tpadjNorm[i]

# negative sampling using pre-sampled entities (preSamp) for efficiency
def negSamp(temLabel, preSamp, sampSize=1000):
	negset = [None] * sampSize
	cur = 0
	for temval in preSamp:
		if temLabel[temval] == 0:
			negset[cur] = temval
			cur += 1
		if cur == sampSize:
			break
	negset = np.array(negset[:cur])
	return negset

def transpose(mat):
	coomat = sp.coo_matrix(mat)
	return csr_matrix(coomat.transpose())

def transToLsts(mat, mask=False):
	shape = [mat.shape[0], mat.shape[1]]
	coomat = sp.coo_matrix(mat)
	indices = np.array(list(map(list, zip(coomat.row, coomat.col))), dtype=np.int32)
	data = coomat.data.astype(np.float32)

	# half mask
	if mask:
		spMask = (np.random.uniform(size=data.shape) > 0.5) * 1.0
		data = data * spMask

	if indices.shape[0] == 0:
		indices = np.array([[0, 0]], dtype=np.int32)
		data = np.array([0.0], np.float32)
	return indices, data, shape

def sampleLargeGraph_forJD(pckUsrs, sampDepth=2, sampNum=25000):
	global adjs
	global adj
	global tpadj

	def makeMask(nodes, size):
		mask = np.ones(size)
		if not nodes is None:
			mask[nodes] = 0.0
		return mask

	def updateBdgt(adj, nodes):
		if nodes is None:
			return 0
		tembat = 1000
		ret = 0
		for i in range(int(np.ceil(len(nodes) / tembat))):
			st = tembat * i
			ed = min((i+1) * tembat, len(nodes))
			temNodes = nodes[st: ed]
			ret += np.sum(adj[temNodes], axis=0)
		return ret

	def updateUsrBdgt(adj, ItmBudget):
		return np.sum(adj.multiply(np.reshape(ItmBudget, [-1, 1])), axis=0)

	def sample(budget, mask, sampNum):
		score = (mask * np.reshape(np.array(budget), [-1])) ** 2
		norm = np.sum(score)
		if norm == 0:
			return np.random.choice(len(score), 1)
		score = list(score / norm)
		pckNodes = np.random.choice(len(score), sampNum, p=score, replace=False)
		return pckNodes

	usrMask = makeMask(pckUsrs, adj.shape[0])
	itmBdgt = updateBdgt(adj, pckUsrs)
	usrBdgt = updateUsrBdgt(tpadj, itmBdgt)
	for i in range(sampDepth):
		newUsrs = sample(usrBdgt, usrMask, sampNum)
		usrMask = usrMask * makeMask(newUsrs, adj.shape[0])
		if i == sampDepth - 1:
			break
		itmBdgt += updateBdgt(adj, newUsrs)
		usrBdgt += updateUsrBdgt(tpadj, itmBdgt)
	usrs = np.reshape(np.argwhere(usrMask==0), [-1])

	pckAdjs = []
	pckTpAdjs = []
	for i in range(len(adjs)):
		pckU = adjs[i][usrs]
		tpPckI = transpose(pckU)
		pckTpAdjs.append(tpPckI)
		pckAdjs.append(pckU)
	return pckAdjs, pckTpAdjs, usrs

def sampleLargeGraph(pckUsrs, pckItms=None, sampDepth=2, sampNum=62500):
	global adjs
	global adj
	global tpadj

	def makeMask(nodes, size):
		mask = np.ones(size)
		if not nodes is None:
			mask[nodes] = 0.0
		return mask

	def updateBdgt(adj, nodes):
		if nodes is None:
			return 0
		tembat = 1000
		ret = 0
		for i in range(int(np.ceil(len(nodes) / tembat))):
			st = tembat * i
			ed = min((i+1) * tembat, len(nodes))
			temNodes = nodes[st: ed]
			ret += np.sum(adj[temNodes], axis=0)
		return ret

	def sample(budget, mask, sampNum):
		score = (mask * np.reshape(np.array(budget), [-1])) ** 2
		norm = np.sum(score)
		if norm == 0:
			return np.random.choice(len(score), 1)
		score = list(score / norm)
		pckNodes = np.random.choice(len(score), sampNum, p=score, replace=False)
		return pckNodes

	usrMask = makeMask(pckUsrs, adj.shape[0])
	itmMask = makeMask(pckItms, adj.shape[1])
	itmBdgt = updateBdgt(adj, pckUsrs)
	if pckItms is None:
		pckItms = sample(itmBdgt, itmMask, len(pckUsrs))
		itmMask = itmMask * makeMask(pckItms, adj.shape[1])
	usrBdgt = updateBdgt(tpadj, pckItms)
	for i in range(sampDepth):
		newUsrs = sample(usrBdgt, usrMask, sampNum)
		usrMask = usrMask * makeMask(newUsrs, adj.shape[0])
		newItms = sample(itmBdgt, itmMask, sampNum)
		itmMask = itmMask * makeMask(newItms, adj.shape[1])
		if i == sampDepth - 1:
			break
		usrBdgt += updateBdgt(tpadj, newItms)
		itmBdgt += updateBdgt(adj, newUsrs)
	usrs = np.reshape(np.argwhere(usrMask==0), [-1])
	itms = np.reshape(np.argwhere(itmMask==0), [-1])
	pckAdjs = []
	pckTpAdjs = []
	for i in range(len(adjs)):
		pckU = adjs[i][usrs]
		tpPckI = transpose(pckU)[itms]
		pckTpAdjs.append(tpPckI)
		pckAdjs.append(transpose(tpPckI))
	return pckAdjs, pckTpAdjs, usrs, itms

def sampleNodes(pckUsrs, pckItms=None, sampDepth=3, sampNum=10):
	global adjs
	global adj
	global tpadj

	def makeMask(nodes, size):
		mask = np.ones(size)
		if not nodes is None:
			mask[nodes] = 0.0
		return mask

	def updateBdgt(adj, nodes):
		if nodes is None:
			return 0
		tembat = 1000
		ret = 0
		for i in range(int(np.ceil(len(nodes) / tembat))):
			st = tembat * i
			ed = min((i+1) * tembat, len(nodes))
			temNodes = nodes[st: ed]
			ret += np.sum(adj[temNodes], axis=0)
		return ret

	def sample(budget, mask, sampNum):
		score = (mask * np.reshape(np.array(budget), [-1])) ** 2
		norm = np.sum(score)
		if norm == 0:
			return np.random.choice(len(score), 1)
		score = list(score / norm)
		pckNodes = np.random.choice(len(score), sampNum, p=score, replace=False)
		return pckNodes

	usrMask = makeMask(pckUsrs, adj.shape[0])
	itmMask = makeMask(pckItms, adj.shape[1])
	itmBdgt = updateBdgt(adj, pckUsrs)
	if pckItms is None:
		pckItms = sample(itmBdgt, itmMask, len(pckUsrs))
		itmMask = itmMask * makeMask(pckItms, adj.shape[1])
	usrBdgt = updateBdgt(tpadj, pckItms)
	uNum = len(pckUsrs)
	iNum = len(pckItms)
	for i in range(sampDepth):
		newUsrs = sample(usrBdgt, usrMask, min(sampNum * iNum, 100))
		usrMask = usrMask * makeMask(newUsrs, adj.shape[0])
		newItms = sample(itmBdgt, itmMask, min(sampNum * uNum, 100))
		itmMask = itmMask * makeMask(newItms, adj.shape[1])
		uNum = len(newUsrs)
		iNum = len(newItms)
		if i == sampDepth - 1:
			break
		usrBdgt += updateBdgt(tpadj, newItms)
		itmBdgt += updateBdgt(adj, newUsrs)
	usrs = np.reshape(np.argwhere(usrMask==0), [-1])
	itms = np.reshape(np.argwhere(itmMask==0), [-1])
	pckAdjs = []
	pckTpAdjs = []
	for i in range(len(adjs)):
		pckU = adjs[i][usrs]
		tpPckI = transpose(pckU)[itms]
		pckTpAdjs.append(tpPckI)
		pckAdjs.append(transpose(tpPckI))
	return pckAdjs, pckTpAdjs, usrs, itms
