package com.example;

import java.util.ArrayList;
import java.util.List;

/**
 * Splits one refund across the line items it covers.
 *
 * Everything here is integer cents. A refund split has to add back up to the
 * amount that was refunded -- that is what reconciliation checks the next
 * morning -- and no rounding scheme over fractional amounts gives that for
 * free. The remainder is handed out one cent at a time instead, from the
 * first item onwards, so two runs over the same order produce the same split.
 */
public class RefundSplitter {

    /**
     * @param refundCents the amount refunded, in cents; never negative and
     *                    never more than the items add up to
     * @param itemCents   each line item's amount, in cents
     * @return one share per item, in the same order, summing to refundCents
     */
    public List<Long> split(long refundCents, List<Long> itemCents) {
        if (refundCents < 0) {
            throw new IllegalArgumentException("退款金额不能为负");
        }
        long total = 0;
        for (Long item : itemCents) {
            if (item == null || item < 0) {
                throw new IllegalArgumentException("明细金额不能为负");
            }
            total = Math.addExact(total, item);
        }
        if (refundCents > total) {
            throw new IllegalArgumentException("退款金额不能超过明细合计");
        }
        List<Long> shares = new ArrayList<>(itemCents.size());
        if (total == 0) {
            for (int index = 0; index < itemCents.size(); index++) {
                shares.add(0L);
            }
            return shares;
        }
        long assigned = 0;
        for (Long item : itemCents) {
            // multiplyExact rather than a silent wrap: a split that overflowed
            // is not a split that is slightly wrong, it is one nobody can
            // reconcile, and it should stop here rather than reach the ledger.
            long share = Math.multiplyExact(refundCents, item) / total;
            shares.add(share);
            assigned += share;
        }
        long remainder = refundCents - assigned;
        for (int index = 0; index < shares.size() && remainder > 0; index++) {
            shares.set(index, shares.get(index) + 1);
            remainder--;
        }
        return shares;
    }
}
