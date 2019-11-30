import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from experiment.utils import *
from nltk.translate.bleu_score import corpus_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print_freq = 100

def validate(val_loader, decoder, criterion_ce, criterion_dis, word_map=None):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    with torch.no_grad(): 
        for i, (imgs, caps, caplens,allcaps) in enumerate(val_loader):
            
            #check
            
            # Move to device, if available
            imgs = imgs.to(device)
            #caps = caps.to(device)
            caps = [c.to(device) for c in caps]
            caplens = caplens.to(device)

            scores, scores_d, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)
            
            #Max-pooling across predicted words across time steps for discriminative supervision
            scores_d = scores_d.max(1)[0]

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]
            targets_d = torch.zeros(scores_d.size(0),scores_d.size(1)).to(device)
            targets_d.fill_(-1)

            for length in decode_lengths:
                targets_d[:,:length-1] = targets[:,:length-1]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss_d = criterion_dis(scores_d,targets_d.long())
            loss_g = criterion_ce(scores, targets)
            loss = loss_g + (10 * loss_d)

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            #allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            #위 함수 구현
            allcaps_sorted = []

            
            for i in sort_ind:
                allcaps_sorted.append(allcaps[0][i.item()].tolist())
            #allcaps = allcaps_sorted.to(device)
            allcaps = allcaps_sorted
                   
            PAD_IDX = 0
            CLS_IDX = 101
            SEP_IDX = 102
            
            for j in range(len(allcaps)):
                img_caps = allcaps[j] #[101 253 23 123 231]
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {PAD_IDX, CLS_IDX}],#또 에러나면 SEP_IDX 제거
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)
    bleu4 = round(bleu4,4)

    print(
        '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses,
            top5=top5accs,
            bleu=bleu4))

    return bleu4